"""
backend-api/app/telemetry.py
-----------------------------
OpenTelemetry bootstrap for the Backend API.

This module is the single place that owns all telemetry setup so that
main.py stays focused on business logic.  Import and call setup_telemetry()
once, before the FastAPI app is created.

What this module does
~~~~~~~~~~~~~~~~~~~~~
1. TracerProvider
   - Reads OTEL_SERVICE_NAME / OTEL_RESOURCE_ATTRIBUTES from env.
   - Sends spans via OTLP/gRPC or OTLP/HTTP depending on
     OTEL_EXPORTER_OTLP_PROTOCOL (defaults to grpc).
   - Falls back to a ConsoleSpanExporter when OTEL_EXPORTER_OTLP_ENDPOINT
     is not set, so spans are always visible during development even
     without a collector.

2. MeterProvider
   - Same resource and endpoint story as the tracer provider.
   - Falls back to a ConsoleMetricExporter when no endpoint is set.
   - Creates the application-level instruments (counter, histogram, …)
     that main.py uses.

3. Logging enrichment
   - Installs a logging filter that reads the current OTel context and
     appends trace_id / span_id to every log record.
   - This is the lightweight approach: we keep stdlib logging and just
     enrich the records rather than replacing the entire logging backend
     with OTel's LoggerProvider.  That is a deliberate tradeoff — see
     the note at the bottom of this file.

Auto-instrumentation vs manual instrumentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AUTO  (wired inside setup_telemetry via instrumentors):
  - FastAPIInstrumentor  — every HTTP request gets a server span automatically.
  - All standard HTTP semantic conventions are filled in for you.

MANUAL (called explicitly in main.py):
  - Custom "datastore.query" child span in the /data endpoint.
  - Manual metric increments for request_count, error_count, in_flight.
  - These teach you how instrumentation works at the SDK level.

High-cardinality label danger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We only use these attributes on metrics:
  http.method, http.route, http.status_code, service.name
NEVER add request_id, user_id, session_id, or IP addresses to metrics.
Each unique combination of label values creates a new time-series in
Prometheus.  A million unique user IDs = a million time-series = OOM.
Traces are the right tool for per-request identity; metrics are for
aggregates.

Common beginner mistakes
~~~~~~~~~~~~~~~~~~~~~~~~
1. Calling setup_telemetry() AFTER FastAPIInstrumentor().instrument(app=…).
   The instrumentor must see the providers that were registered globally first.
   Always call setup_telemetry() before creating the FastAPI app.

2. Forgetting to flush on shutdown.  Spans and metrics are batched in memory.
   If the process exits before the batch is sent, you lose the last seconds
   of data.  The on_shutdown handler in main.py calls force_flush + shutdown.

3. Using the default BatchSpanProcessor export interval (5 s) and then
   wondering why spans "disappear" — they are just waiting in the buffer.

4. Putting high-cardinality values like user IDs on Histogram attributes.
   Histograms already have many time-series per bucket; multiplying by
   millions of user IDs will crash your metrics backend.
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

logger = logging.getLogger("backend-api.telemetry")

# ── Module-level instrument handles ───────────────────────────────────────────
# Stored here so main.py can import them directly without touching providers.

# Total HTTP requests handled (labels: http.method, http.route, http.status_code)
request_counter: Optional[metrics.Counter] = None

# Requests that ended with HTTP >= 500
error_counter: Optional[metrics.Counter] = None

# Request latency in seconds (same labels as request_counter)
latency_histogram: Optional[metrics.Histogram] = None

# Currently in-flight requests
in_flight_gauge: Optional[metrics.UpDownCounter] = None

# Total fake-datastore queries executed
db_query_counter: Optional[metrics.Counter] = None


# ──────────────────────────────────────────────────────────────────────────────
# Logging enrichment
# ──────────────────────────────────────────────────────────────────────────────

class OtelContextFilter(logging.Filter):
    """
    Injects the active OTel trace_id and span_id into every log record.

    Usage in log format:
        %(otel_trace_id)s  %(otel_span_id)s

    Tradeoff note
    ~~~~~~~~~~~~~
    This approach keeps stdlib logging and just decorates records.
    The "full" OTel approach uses opentelemetry-sdk's LoggerProvider and
    LogRecord exporters to ship structured log events over OTLP.  That is
    more powerful (log-trace correlation in the backend without parsing)
    but also heavier and harder to learn first.  We start here so you can
    see trace_id appear in terminal logs immediately.  When you add Loki
    later, you can either parse the trace_id from the log string or
    upgrade to the full OTEL logging SDK.
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
    """Attach the OtelContextFilter to the root logger."""
    root = logging.getLogger()
    # Update the formatter to include the new fields.
    # Using a format that is easy to parse with Loki / regex later.
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
    """
    Build the OTel Resource that identifies this service.

    The SDK automatically reads:
        OTEL_SERVICE_NAME          (e.g. "backend-api")
        OTEL_RESOURCE_ATTRIBUTES   (e.g. "deployment.environment=local,team=sre")

    We also add a hard-coded service.version so every span and metric carries it.
    """
    return Resource.create(
        {
            "service.name":    os.getenv("OTEL_SERVICE_NAME",    "backend-api"),
            "service.version": os.getenv("SERVICE_VERSION",      "1.0.0"),
        }
    )


# ──────────────────────────────────────────────────────────────────────────────
# Exporter factories
# ──────────────────────────────────────────────────────────────────────────────

def _make_span_exporter():
    """
    Return the right span exporter based on environment variables.

    Priority order:
      1. If OTEL_EXPORTER_OTLP_ENDPOINT is set:
           - Use gRPC if OTEL_EXPORTER_OTLP_PROTOCOL == "grpc" (default)
           - Use HTTP/protobuf if protocol == "http/protobuf"
      2. No endpoint → ConsoleSpanExporter (prints to stdout/stderr)

    The console exporter is invaluable during development: you can see
    exactly what span data you are generating without running a collector.
    """
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc").lower()

    if endpoint:
        if protocol == "http/protobuf":
            logger.info("Trace exporter: OTLP/HTTP  endpoint=%s", endpoint)
            return HTTPSpanExporter()   # reads OTEL_EXPORTER_OTLP_ENDPOINT automatically
        else:
            logger.info("Trace exporter: OTLP/gRPC  endpoint=%s", endpoint)
            return GRPCSpanExporter()   # reads OTEL_EXPORTER_OTLP_ENDPOINT automatically

    logger.warning(
        "OTEL_EXPORTER_OTLP_ENDPOINT not set — using ConsoleSpanExporter. "
        "Spans will appear in stdout. Set the endpoint to send to a collector."
    )
    return ConsoleSpanExporter()


def _make_metric_exporter():
    """Same logic as _make_span_exporter but for metrics."""
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc").lower()

    if endpoint:
        if protocol == "http/protobuf":
            logger.info("Metric exporter: OTLP/HTTP  endpoint=%s", endpoint)
            return HTTPMetricExporter()
        else:
            logger.info("Metric exporter: OTLP/gRPC  endpoint=%s", endpoint)
            return GRPCMetricExporter()

    logger.warning(
        "OTEL_EXPORTER_OTLP_ENDPOINT not set — using ConsoleMetricExporter. "
        "Metrics will appear in stdout every 30 s."
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
    Bootstrap all OpenTelemetry providers and instruments.

    Call this exactly ONCE, at the top of main.py, BEFORE creating any
    routes or middleware.

    Parameters
    ----------
    fastapi_app:
        The FastAPI() application instance.  Passed to FastAPIInstrumentor
        so auto-instrumentation can wrap it.
    """
    global request_counter, error_counter, latency_histogram, in_flight_gauge, db_query_counter

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
    logger.info("MeterProvider configured  export_interval_ms=%s",
                os.getenv("OTEL_METRIC_EXPORT_INTERVAL_MS", "15000"))

    # ── Auto-instrumentation — FastAPI ────────────────────────────────────────
    # FastAPIInstrumentor automatically creates a SERVER span for every
    # incoming HTTP request.  It fills in all standard semantic-convention
    # attributes: http.method, http.route, http.status_code, net.host.name …
    # You do NOT need to write any span code in your route handlers for this.
    FastAPIInstrumentor.instrument_app(
        fastapi_app,
        tracer_provider=tracer_provider,
        # Exclude health endpoint from traces to reduce noise.
        # Liveness probes fire every few seconds and create useless spans.
        excluded_urls="/health",
    )
    logger.info("FastAPIInstrumentor attached — all routes auto-instrumented (except /health)")

    # ── Manual metrics instruments ────────────────────────────────────────────
    # These are created AFTER the MeterProvider is registered so they use
    # the correct provider rather than the no-op default.
    meter = metrics.get_meter("backend-api", version="1.0.0")

    request_counter = meter.create_counter(
        name="http_requests_total",
        description="Total number of HTTP requests handled by this service.",
        unit="1",
    )
    error_counter = meter.create_counter(
        name="http_errors_total",
        description="Total number of HTTP requests that resulted in a 5xx response.",
        unit="1",
    )
    latency_histogram = meter.create_histogram(
        name="http_request_duration_seconds",
        description="End-to-end request latency measured inside the route handler.",
        unit="s",
    )
    in_flight_gauge = meter.create_up_down_counter(
        name="http_requests_in_flight",
        description="Number of HTTP requests currently being processed.",
        unit="1",
    )
    db_query_counter = meter.create_counter(
        name="datastore_queries_total",
        description="Total number of fake-datastore queries executed.",
        unit="1",
    )

    logger.info("Manual metric instruments created: "
                "http_requests_total, http_errors_total, "
                "http_request_duration_seconds, http_requests_in_flight, "
                "datastore_queries_total")

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
    """
    Convenience accessor — returns a named tracer for this service.

    Use this in main.py when you want to create a custom child span:

        from app.telemetry import get_tracer
        tracer = get_tracer()

        with tracer.start_as_current_span("my.operation") as span:
            span.set_attribute("key", "value")
            ...
    """
    return trace.get_tracer("backend-api", version="1.0.0")
