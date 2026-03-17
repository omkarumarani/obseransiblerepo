"""
backend-api/app/main.py
-----------------------
Backend API service for the Observability Learning project.

This version adds OpenTelemetry instrumentation while keeping the same
endpoint behaviour as v1.  Every change is explained with a comment so you
can see exactly what instrumentation does and why.

Endpoints
~~~~~~~~~
  GET /health  — liveness probe   (excluded from traces to reduce noise)
  GET /ok      — normal success
  GET /slow    — artificial delay (BACKEND_SLOW_MS ms)
  GET /error   — always HTTP 500
  GET /data    — fake datastore read with a custom child span

Telemetry emitted
~~~~~~~~~~~~~~~~~
  Traces  — auto via FastAPIInstrumentor (every request except /health)
           + manual "datastore.query" child span on /data
  Metrics — http_requests_total, http_errors_total,
            http_request_duration_seconds, http_requests_in_flight,
            datastore_queries_total  (all incremented manually)
  Logs    — every log record enriched with trace_id and span_id via
            the OtelContextFilter installed by setup_telemetry()
"""

import asyncio
import logging
import os
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from opentelemetry import trace
from opentelemetry.trace import StatusCode  # noqa: F401  (available for custom use)

# ── Telemetry bootstrap ────────────────────────────────────────────────────────
# setup_telemetry() must be called BEFORE creating the FastAPI app so that
# FastAPIInstrumentor can wrap the app at creation time, not after.
import app.telemetry as _tel
from app.telemetry import get_tracer, setup_telemetry

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# The format is upgraded by telemetry.py's _install_log_enrichment() to include
# trace_id and span_id once setup_telemetry() is called.
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("backend-api")


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
SLOW_MS: int = int(os.getenv("BACKEND_SLOW_MS", "1500"))
logger.info("Backend configuration: SLOW_MS=%d", SLOW_MS)


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# IMPORTANT: setup_telemetry(app) is called immediately after app is created.
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Backend API",
    version="1.0.0",
    description="Backend service for the Observability Learning project.",
)

# Wire up all OTel providers and auto-instrumentors.
# After this call:
#   - Every request gets a server span (FastAPIInstrumentor).
#   - Metric instruments are initialised.
#   - Log format includes trace_id and span_id.
setup_telemetry(app)


# ──────────────────────────────────────────────────────────────
# In-memory datastore
#
# Simulates a real database without any external dependency.
# The dict structure makes it easy to:
#   - add artificial latency (simulate slow queries)
#   - inject errors (simulate DB connection failures)
#   - track query counts (expose as a Prometheus counter later)
# ──────────────────────────────────────────────────────────────
_DATASTORE: dict = {
    "items": [
        {"id": 1, "name": "Widget Alpha",     "stock": 42,  "category": "widgets"},
        {"id": 2, "name": "Gadget Beta",       "stock": 7,   "category": "gadgets"},
        {"id": 3, "name": "Doohickey Gamma",   "stock": 100, "category": "misc"},
        {"id": 4, "name": "Thingamajig Delta", "stock": 0,   "category": "misc"},
        {"id": 5, "name": "Widget Epsilon",    "stock": 15,  "category": "widgets"},
    ],
    # Tracks total queries across the lifetime of this process.
    # Natural candidate for a Prometheus counter later.
    "query_count": 0,
}


# ──────────────────────────────────────────────────────────────────────────────
# Middleware — metrics recording + request logging
#
# FastAPIInstrumentor already creates a span for each request.
# This middleware records the MANUAL metrics (counter, histogram) and
# logs the request.  Using middleware keeps the route handlers clean.
#
# Why manual metrics alongside auto-instrumentation?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FastAPIInstrumentor records spans (traces).  It does NOT automatically
# write to YOUR MeterProvider.  You must add metric recording explicitly.
# This is a very common point of confusion for beginners.
# ──────────────────────────────────────────────────────────────────────────────
@app.middleware("http")
async def record_metrics_and_log(request: Request, call_next):
    # Labels for all metrics on this request.
    # Only low-cardinality values — never put user IDs or request IDs here.
    attrs = {
        "http.method":  request.method,
        "http.route":   request.url.path,
        "service.name": "backend-api",
    }

    # Track how many requests are currently in-flight.
    _tel.in_flight_gauge.add(1, attrs)

    start = time.perf_counter()
    response = await call_next(request)
    duration_s = time.perf_counter() - start

    # Add status code AFTER the response is ready.
    attrs_with_status = {**attrs, "http.status_code": str(response.status_code)}

    _tel.request_counter.add(1, attrs_with_status)
    _tel.latency_histogram.record(duration_s, attrs_with_status)
    _tel.in_flight_gauge.add(-1, attrs)      # decrement — request is done

    if response.status_code >= 500:
        _tel.error_counter.add(1, attrs_with_status)

    logger.info(
        "request  method=%s  path=%s  status=%d  duration_ms=%.1f",
        request.method,
        request.url.path,
        response.status_code,
        duration_s * 1000,
    )
    return response


# ──────────────────────────────────────────────────────────────────────────────
# Shutdown — flush telemetry buffers
#
# OTel uses a BatchSpanProcessor that holds spans in memory until a batch
# is ready to send.  If the process exits before the flush, the last spans
# are lost.  This handler gives the processor a chance to drain.
# ──────────────────────────────────────────────────────────────────────────────
@app.on_event("shutdown")
async def shutdown_telemetry():
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()
        logger.info("TracerProvider flushed on shutdown")


# ──────────────────────────────────────────────────────────────────────────────
# GET /health
# Excluded from traces by FastAPIInstrumentor (excluded_urls="/health").
# Liveness probes are noisy and rarely interesting in a trace.
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
async def health():
    return {"service": "backend-api", "status": "healthy"}


# ──────────────────────────────────────────────────────────────────────────────
# GET /ok
#
# Telemetry emitted
# ~~~~~~~~~~~~~~~~~
# Auto trace: SERVER span "GET /ok" with http.status_code=200
# Manual metric: http_requests_total +1, http_request_duration_seconds record
# Log: INFO with trace_id and span_id printed inline
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/ok", tags=["demo"])
async def ok():
    logger.info("Handling /ok — returning normal response")
    return {
        "service": "backend-api",
        "endpoint": "/ok",
        "status": "success",
        "message": "Backend is operating normally.",
    }


# ──────────────────────────────────────────────────────────────────────────────
# GET /slow
#
# Telemetry emitted
# ~~~~~~~~~~~~~~~~~
# Auto trace: SERVER span "GET /slow" — span duration will be ~BACKEND_SLOW_MS
#             This is the key learning moment: the span width in Tempo
#             directly corresponds to the sleep duration you set.
# Manual metric: http_request_duration_seconds will show a high-percentile spike.
#                Watch p95/p99 rise in Grafana after a few calls here.
# Log: INFO with trace_id/span_id, making it possible to jump from log → trace.
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/slow", tags=["demo"])
async def slow():
    delay_s = SLOW_MS / 1000.0
    logger.info("Handling /slow — sleeping %.3fs (%dms)", delay_s, SLOW_MS)
    await asyncio.sleep(delay_s)
    return {
        "service": "backend-api",
        "endpoint": "/slow",
        "status": "success",
        "message": f"Response intentionally delayed by {SLOW_MS}ms.",
        "delay_ms": SLOW_MS,
    }


# ──────────────────────────────────────────────────────────────────────────────
# GET /error
#
# Telemetry emitted
# ~~~~~~~~~~~~~~~~~
# Auto trace: SERVER span "GET /error"
#             FastAPIInstrumentor sets span status=ERROR for 5xx responses.
# Manual metric: http_errors_total +1 (recorded in middleware above)
# Log: WARNING with trace_id/span_id — easy to query in Loki:
#      {service="backend-api"} |= "WARNING"
#
# Learning moment: in Tempo, this span is RED.  In Grafana the error_rate
# graph spikes.  In Loki the WARNING log appears.  All three point to the
# same trace_id — that is the M-E-L correlation you are building toward.
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/error", tags=["demo"])
async def error():
    logger.warning("Handling /error — returning simulated HTTP 500")
    return JSONResponse(
        status_code=500,
        content={
            "service": "backend-api",
            "endpoint": "/error",
            "status": "error",
            "message": "Simulated internal backend error.",
            "error_code": "SIMULATED_BACKEND_FAILURE",
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /data
#
# This endpoint demonstrates MANUAL span creation — a child span nested
# inside the auto-created server span.
#
# Span hierarchy in a trace viewer:
#   [SERVER span: GET /data  ──────────────────────────────────────────────]
#     [INTERNAL span: datastore.query  ──────────────────────────────]
#
# The child span has custom attributes:
#   db.system        = "in-memory"
#   db.operation     = "scan"
#   db.result.total  = 5
#   db.result.in_stock     = 4
#   db.result.out_of_stock = 1
#
# These attributes are the OTel equivalent of what you would add to a real
# DB client span.  They make it trivial to answer "how many rows did this
# query return?" just by looking at the trace.
#
# Telemetry emitted
# ~~~~~~~~~~~~~~~~~
# Auto trace:   SERVER span "GET /data"
# Manual trace: INTERNAL child span "datastore.query"
# Manual metric: datastore_queries_total +1
# Manual metric: http_requests_total, latency, in_flight (from middleware)
# Log: INFO with trace_id/span_id
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/data", tags=["demo"])
async def data():
    tracer = get_tracer()

    # Start a child span.  Because this runs inside the FastAPIInstrumentor
    # server span, it is automatically a child of that span — no manual
    # parent linking needed.  The OTel context propagation handles this.
    with tracer.start_as_current_span("datastore.query") as db_span:
        # Record semantic-convention attributes on the span.
        db_span.set_attribute("db.system",    "in-memory")
        db_span.set_attribute("db.operation", "scan")

        # Simulate datastore work
        _DATASTORE["query_count"] += 1
        query_id = _DATASTORE["query_count"]
        items    = _DATASTORE["items"]
        in_stock     = [i for i in items if i["stock"] > 0]
        out_of_stock = [i for i in items if i["stock"] == 0]

        # Add result attributes — very useful for debugging in Tempo
        db_span.set_attribute("db.result.total",         len(items))
        db_span.set_attribute("db.result.in_stock",      len(in_stock))
        db_span.set_attribute("db.result.out_of_stock",  len(out_of_stock))
        db_span.set_attribute("db.query_id",             query_id)

        # Increment the datastore query counter metric
        _tel.db_query_counter.add(1, {
            "db.operation": "scan",
            "service.name": "backend-api",
        })

        logger.info(
            "Datastore read  query_id=%d  total=%d  in_stock=%d  out_of_stock=%d",
            query_id, len(items), len(in_stock), len(out_of_stock),
        )

    return {
        "service": "backend-api",
        "endpoint": "/data",
        "status": "success",
        "query_id": query_id,
        "total_items": len(items),
        "in_stock_count": len(in_stock),
        "out_of_stock_count": len(out_of_stock),
        "items": items,
    }
