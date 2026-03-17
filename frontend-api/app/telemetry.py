"""
frontend-api/app/telemetry.py
------------------------------
OpenTelemetry bootstrap for the Frontend API.

This is the mirror of backend-api/app/telemetry.py with two additions:
  1. HTTPXClientInstrumentor — auto-instruments every outgoing httpx call
     so W3C traceparent headers are injected automatically.  This is what
     makes cross-service context propagation work without any manual header
     code in _call_backend().
  2. An extra "downstream_call" histogram that tracks how long outbound
     calls to the backend take, independently of the overall request latency.
     This teaches the concept of decomposing a request into its parts.

What is auto-instrumented here
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  - FastAPIInstrumentor: server spans for every inbound request.
  - HTTPXClientInstrumentor: client spans for every outbound httpx request,
    PLUS automatic traceparent / tracestate header injection into the
    outgoing request.  The backend receives the header and connects its
    server span to this client span — that is the full distributed trace.

What must still be done manually
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  - Custom child span in _call_backend() for additional context/attributes
    beyond what HTTPXClientInstrumentor provides.
  - Metric increments (request_counter, error_counter, latency_histogram).
  - Marking spans as ERROR when a 5xx is received from the backend.

See backend-api/app/telemetry.py for the full set of learning notes
on high-cardinality dangers and common beginner mistakes.
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
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter as HTTPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPSpanExporter,
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
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter as GRPCLogExporter,
)
from opentelemetry.exporter.otlp.proto.http._log_exporter import (
    OTLPLogExporter as HTTPLogExporter,
)

logger = logging.getLogger("frontend-api.telemetry")

# ── Module-level instrument handles ───────────────────────────────────────────

request_counter: Optional[metrics.Counter]        = None
error_counter: Optional[metrics.Counter]          = None
latency_histogram: Optional[metrics.Histogram]    = None
in_flight_gauge: Optional[metrics.UpDownCounter]  = None
downstream_histogram: Optional[metrics.Histogram] = None


# ──────────────────────────────────────────────────────────────────────────────
# Logging enrichment  (identical pattern to backend)
# ──────────────────────────────────────────────────────────────────────────────

class OtelContextFilter(logging.Filter):
    """
    Injects trace_id and span_id into every stdlib log record.

    This lightweight approach means you can grep your terminal logs for a
    trace_id and then look that same ID up in Tempo to jump straight to
    the trace.  That is log-trace correlation in its simplest form.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        span = trace.get_current_span()
        ctx  = span.get_span_context()
        if ctx.is_valid:
            record.otel_trace_id = format(ctx.trace_id, "032x")
            record.otel_span_id  = format(ctx.span_id,  "016x")
        else:
            record.otel_trace_id = "0" * 32
            record.otel_span_id  = "0" * 16
        return True


def _install_log_enrichment() -> None:
    root = logging.getLogger()
    fmt = (
        "%(asctime)s [%(levelname)s] %(name)s "
        "trace_id=%(otel_trace_id)s span_id=%(otel_span_id)s "
        "- %(message)s"
    )
    for handler in root.handlers:
        handler.setFormatter(logging.Formatter(fmt))
    root.addFilter(OtelContextFilter())
    logger.debug("OTel log enrichment installed")


# ──────────────────────────────────────────────────────────────────────────────
# Resource
# ──────────────────────────────────────────────────────────────────────────────

def _build_resource() -> Resource:
    return Resource.create(
        {
            "service.name":    os.getenv("OTEL_SERVICE_NAME",    "frontend-api"),
            "service.version": os.getenv("SERVICE_VERSION",      "1.0.0"),
        }
    )


# ──────────────────────────────────────────────────────────────────────────────
# Exporter factories  (same as backend, duplicated here intentionally —
# each service owns its own telemetry bootstrap, keeping them independently
# deployable and easy to understand in isolation)
# ──────────────────────────────────────────────────────────────────────────────

def _make_span_exporter():
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc").lower()
    if endpoint:
        if protocol == "http/protobuf":
            logger.info("Trace exporter: OTLP/HTTP  endpoint=%s", endpoint)
            return HTTPSpanExporter()
        logger.info("Trace exporter: OTLP/gRPC  endpoint=%s", endpoint)
        return GRPCSpanExporter()
    logger.warning(
        "OTEL_EXPORTER_OTLP_ENDPOINT not set — using ConsoleSpanExporter."
    )
    return ConsoleSpanExporter()


def _make_metric_exporter():
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc").lower()
    if endpoint:
        if protocol == "http/protobuf":
            logger.info("Metric exporter: OTLP/HTTP  endpoint=%s", endpoint)
            return HTTPMetricExporter()
        logger.info("Metric exporter: OTLP/gRPC  endpoint=%s", endpoint)
        return GRPCMetricExporter()
    logger.warning(
        "OTEL_EXPORTER_OTLP_ENDPOINT not set — using ConsoleMetricExporter."
    )
    return ConsoleMetricExporter()


def _make_log_exporter():
    """
    Return the right OTLP log exporter based on environment variables.

    Falls back to None when no endpoint is set — in that case the
    LoggerProvider is not configured and logs are not exported via OTLP.
    They still appear on the terminal via the stdlib StreamHandler.
    """
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc").lower()
    if endpoint:
        if protocol == "http/protobuf":
            logger.info("Log exporter: OTLP/HTTP  endpoint=%s", endpoint)
            return HTTPLogExporter()
        logger.info("Log exporter: OTLP/gRPC  endpoint=%s", endpoint)
        return GRPCLogExporter()
    logger.warning(
        "OTEL_EXPORTER_OTLP_ENDPOINT not set — logs will not be exported via OTLP."
    )
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Main setup entry point
# ──────────────────────────────────────────────────────────────────────────────

def setup_telemetry(fastapi_app) -> None:
    """
    Bootstrap all OpenTelemetry providers and instruments for the frontend.

    Call exactly ONCE at the top of main.py before creating routes.
    """
    global request_counter, error_counter, latency_histogram, in_flight_gauge, downstream_histogram

    resource = _build_resource()

    # ── Tracer provider ───────────────────────────────────────────────────────
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(_make_span_exporter())
    )
    trace.set_tracer_provider(tracer_provider)
    logger.info("TracerProvider configured  service=%s", resource.attributes.get("service.name"))

    # ── Meter provider ────────────────────────────────────────────────────────
    metric_reader = PeriodicExportingMetricReader(
        _make_metric_exporter(),
        export_interval_millis=int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL_MS", "15000")),
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    # ── Auto-instrumentation — FastAPI ────────────────────────────────────────
    FastAPIInstrumentor.instrument_app(
        fastapi_app,
        tracer_provider=tracer_provider,
        excluded_urls="/health",
    )
    logger.info("FastAPIInstrumentor attached")

    # ── Auto-instrumentation — httpx ──────────────────────────────────────────
    # This single call instruments EVERY httpx request made anywhere in this
    # process.  It:
    #   a) Creates a CLIENT span for each outbound request.
    #   b) Injects the W3C traceparent + tracestate headers into the request.
    #
    # Because of (b), when the frontend calls backend-api, the backend receives
    # the traceparent header and its server span is automatically linked as a
    # CHILD of this client span.  The two spans share the same trace_id and
    # appear together in Tempo as a single distributed waterfall — no manual
    # header code needed anywhere.
    HTTPXClientInstrumentor().instrument()
    logger.info("HTTPXClientInstrumentor attached — context propagation enabled")

    # ── Manual metrics instruments ────────────────────────────────────────────
    meter = metrics.get_meter("frontend-api", version="1.0.0")

    request_counter = meter.create_counter(
        name="http_requests_total",
        description="Total HTTP requests handled by this service.",
        unit="1",
    )
    error_counter = meter.create_counter(
        name="http_errors_total",
        description="HTTP requests that resulted in a 5xx response.",
        unit="1",
    )
    latency_histogram = meter.create_histogram(
        name="http_request_duration_seconds",
        description="End-to-end request latency measured inside the route handler.",
        unit="s",
    )
    in_flight_gauge = meter.create_up_down_counter(
        name="http_requests_in_flight",
        description="Requests currently being processed.",
        unit="1",
    )
    # This histogram tracks ONLY the time spent waiting for the backend.
    # Comparing it to latency_histogram teaches you how to decompose latency:
    #   total_latency = frontend_processing + downstream_latency + serialization
    downstream_histogram = meter.create_histogram(
        name="downstream_call_duration_seconds",
        description="Time spent waiting for a response from the backend service.",
        unit="s",
    )

    logger.info("Manual metric instruments created")

    # ── Log enrichment (terminal) ─────────────────────────────────────────────
    # Reformats the existing StreamHandler to include trace_id/span_id text.
    # Called before adding the LoggingHandler so it only reformats the terminal
    # output handler, not the new OTLP handler.
    _install_log_enrichment()

    # ── Logger provider — OTLP log export ─────────────────────────────────────
    # LoggingHandler bridges Python's stdlib logging to OTel LogRecords.
    # Every logger.info/warning/error in this process becomes a structured log
    # record forwarded to the collector's logs pipeline, then to Loki.
    #
    # Each LogRecord carries the current trace_id and span_id automatically —
    # the OTel SDK reads the active span context at emit time.  This is native
    # log-trace correlation: no string parsing needed in Loki.
    log_exporter = _make_log_exporter()
    if log_exporter is not None:
        _loki_logger_provider = LoggerProvider(resource=resource)
        _loki_logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(log_exporter)
        )
        set_logger_provider(_loki_logger_provider)
        logging.getLogger().addHandler(
            LoggingHandler(level=logging.NOTSET, logger_provider=_loki_logger_provider)
        )
        logger.info("LoggerProvider configured — stdlib logs bridged to OTLP")

    logger.info("Telemetry setup complete")


def get_tracer() -> trace.Tracer:
    """Return a named tracer for creating custom spans in main.py."""
    return trace.get_tracer("frontend-api", version="1.0.0")
