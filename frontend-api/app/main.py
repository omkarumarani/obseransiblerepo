"""
frontend-api/app/main.py
------------------------
Frontend API service for the Observability Learning project.

This version adds OpenTelemetry instrumentation.  Endpoint behaviour is
unchanged.  Every instrumentation decision is explained in a comment.

Endpoints
~~~~~~~~~
  GET /health       — liveness probe  (excluded from traces)
  GET /ok           — fast frontend success
  GET /slow         — intentional frontend delay
  GET /error        — HTTP 500, frontend only
  GET /backend-ok   — proxies to backend /ok
  GET /backend-slow — proxies to backend /slow
  GET /backend-error— proxies to backend /error

Telemetry emitted
~~~~~~~~~~~~~~~~~
  Traces  — auto via FastAPIInstrumentor for all inbound requests
           + auto client spans and W3C header injection via HTTPXClientInstrumentor
           + manual "frontend.backend_call" child span in _call_backend()
  Metrics — http_requests_total, http_errors_total,
            http_request_duration_seconds, http_requests_in_flight,
            downstream_call_duration_seconds
  Logs    — every log record enriched with trace_id + span_id
"""

import asyncio
import logging
import os
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from opentelemetry import trace
from opentelemetry.trace import StatusCode

# ── Telemetry bootstrap ────────────────────────────────────────────────────────
import app.telemetry as _tel
from app.telemetry import get_tracer, setup_telemetry

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# The format will be upgraded by telemetry.py's _install_log_enrichment()
# to include trace_id and span_id once setup_telemetry() is called.
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("frontend-api")


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
SLOW_MS: int = int(os.getenv("FRONTEND_SLOW_MS", "800"))
BACKEND_URL: str = os.getenv("BACKEND_URL", "http://backend-api:8081")
logger.info("Frontend configuration: SLOW_MS=%d  BACKEND_URL=%s", SLOW_MS, BACKEND_URL)


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# setup_telemetry(app) wires FastAPIInstrumentor + HTTPXClientInstrumentor.
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Frontend API",
    version="1.0.0",
    description="Frontend service for the Observability Learning project.",
)

setup_telemetry(app)


# ──────────────────────────────────────────────────────────────────────────────
# Middleware — metrics recording + request logging
# ──────────────────────────────────────────────────────────────────────────────
@app.middleware("http")
async def record_metrics_and_log(request: Request, call_next):
    attrs = {
        "http.method":  request.method,
        "http.route":   request.url.path,
        "service.name": "frontend-api",
    }
    _tel.in_flight_gauge.add(1, attrs)
    start = time.perf_counter()
    response = await call_next(request)
    duration_s = time.perf_counter() - start
    attrs_with_status = {**attrs, "http.status_code": str(response.status_code)}
    _tel.request_counter.add(1, attrs_with_status)
    _tel.latency_histogram.record(duration_s, attrs_with_status)
    _tel.in_flight_gauge.add(-1, attrs)
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
# ──────────────────────────────────────────────────────────────────────────────
@app.on_event("shutdown")
async def shutdown_telemetry():
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()
        logger.info("TracerProvider flushed on shutdown")


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper — call the backend service
#
# This is where distributed trace propagation happens.
#
# The HTTPXClientInstrumentor (wired in telemetry.py) automatically:
#   1. Creates a CLIENT span wrapping the httpx.get() call.
#   2. Injects the W3C `traceparent` header into the outgoing request.
#
# The backend receives that header, extracts the trace context, and
# creates its SERVER span as a CHILD of this CLIENT span.  Both spans
# share the same trace_id.  In Tempo they appear as one waterfall.
#
# The manual span below adds frontend-specific attributes on top of what
# HTTPXClientInstrumentor provides, and records the downstream latency
# to the downstream_histogram metric.
# ──────────────────────────────────────────────────────────────────────────────
async def _call_backend(path: str) -> dict:
    url = f"{BACKEND_URL}{path}"
    tracer = get_tracer()

    # Manual span: adds frontend context to the trace that HTTPXClientInstrumentor
    # alone does not provide — e.g. which frontend endpoint triggered this call.
    with tracer.start_as_current_span(
        "frontend.backend_call",
        kind=trace.SpanKind.CLIENT,
    ) as span:
        span.set_attribute("backend.url",  url)
        span.set_attribute("backend.path", path)

        logger.info("Outbound call  GET %s", url)
        ds_start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # HTTPXClientInstrumentor injects traceparent here automatically.
                resp = await client.get(url)
            ds_duration_s = time.perf_counter() - ds_start

            span.set_attribute("http.status_code", resp.status_code)

            # Record downstream latency separately from total request latency.
            # This lets you answer: "is MY service slow or is the backend slow?"
            _tel.downstream_histogram.record(
                ds_duration_s,
                {"backend.path": path, "http.status_code": str(resp.status_code)},
            )

            if resp.status_code >= 500:
                span.set_status(
                    StatusCode.ERROR,
                    f"Backend returned HTTP {resp.status_code}",
                )

            logger.info("Outbound result  GET %s  status=%d", url, resp.status_code)
            return {"status_code": resp.status_code, "body": resp.json()}

        except httpx.RequestError as exc:
            ds_duration_s = time.perf_counter() - ds_start
            _tel.downstream_histogram.record(
                ds_duration_s,
                {"backend.path": path, "http.status_code": "503"},
            )
            # Record the exception on the span so it appears in the trace viewer.
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            logger.error("Backend unreachable  url=%s  error=%s", url, exc)
            return {
                "status_code": 503,
                "body": {
                    "status": "error",
                    "message": f"Could not reach backend service: {exc}",
                    "error_code": "BACKEND_UNREACHABLE",
                },
            }


# ──────────────────────────────────────────────────────────────────────────────
# GET /health
# Excluded from traces (excluded_urls="/health" in FastAPIInstrumentor).
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
async def health():
    return {"service": "frontend-api", "status": "healthy"}


# ──────────────────────────────────────────────────────────────────────────────
# GET /ok
#
# Telemetry emitted
# ~~~~~~~~~~~~~~~~~
# Auto trace: SERVER span "GET /ok",  status=OK
# Manual metric: request_counter +1,  latency_histogram record
# Log: INFO with trace_id and span_id
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/ok", tags=["demo"])
async def ok():
    logger.info("Handling /ok — returning normal response")
    return {
        "service": "frontend-api",
        "endpoint": "/ok",
        "status": "success",
        "message": "Frontend is operating normally.",
    }


# ──────────────────────────────────────────────────────────────────────────────
# GET /slow
#
# Telemetry emitted
# ~~~~~~~~~~~~~~~~~
# Auto trace: SERVER span "GET /slow" — span width = FRONTEND_SLOW_MS
# Manual metric: latency_histogram shows a spike on this route
# Log: INFO with trace_id/span_id
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/slow", tags=["demo"])
async def slow():
    delay_s = SLOW_MS / 1000.0
    logger.info("Handling /slow — sleeping %.3fs (%dms)", delay_s, SLOW_MS)
    await asyncio.sleep(delay_s)
    return {
        "service": "frontend-api",
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
# Auto trace: SERVER span "GET /error" with status=ERROR (5xx triggers it)
# Manual metric: error_counter +1 (in middleware), request_counter +1
# Log: WARNING with trace_id/span_id
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/error", tags=["demo"])
async def error():
    logger.warning("Handling /error — returning simulated HTTP 500")
    return JSONResponse(
        status_code=500,
        content={
            "service": "frontend-api",
            "endpoint": "/error",
            "status": "error",
            "message": "Simulated internal frontend error.",
            "error_code": "SIMULATED_FRONTEND_FAILURE",
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /backend-ok
#
# Telemetry emitted
# ~~~~~~~~~~~~~~~~~
# Auto trace:   SERVER span "GET /backend-ok"  (frontend)
# Manual trace: CLIENT span "frontend.backend_call"  (inside _call_backend)
# Auto trace:   CLIENT span from HTTPXClientInstrumentor  (the actual HTTP call)
# Auto trace:   SERVER span "GET /ok"  (backend — child of above via traceparent)
# Manual metric: downstream_histogram records backend call duration
# Both services log with THE SAME trace_id — this is cross-service correlation.
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/backend-ok", tags=["demo"])
async def backend_ok():
    result = await _call_backend("/ok")
    status_code = result["status_code"]
    return JSONResponse(
        status_code=status_code,
        content={
            "service": "frontend-api",
            "endpoint": "/backend-ok",
            "status": "success" if status_code == 200 else "error",
            "message": "Proxied request to backend /ok.",
            "downstream_result": result["body"],
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /backend-slow
#
# Telemetry emitted
# ~~~~~~~~~~~~~~~~~
# Distributed trace: 3 spans in the waterfall:
#   1. SERVER  frontend "GET /backend-slow"   (wide, same duration as backend)
#   2. CLIENT  frontend "frontend.backend_call"  (child of 1)
#   3. SERVER  backend  "GET /slow"  (~BACKEND_SLOW_MS wide, child of 2)
# Metric: downstream_histogram shows the full backend wait time
# Key lesson: frontend latency_histogram rises even though frontend code is fast.
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/backend-slow", tags=["demo"])
async def backend_slow():
    result = await _call_backend("/slow")
    status_code = result["status_code"]
    return JSONResponse(
        status_code=status_code,
        content={
            "service": "frontend-api",
            "endpoint": "/backend-slow",
            "status": "success" if status_code == 200 else "error",
            "message": "Proxied request to backend /slow.",
            "downstream_result": result["body"],
        },
    )


# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# GET /backend-error
#
# Telemetry emitted
# ~~~~~~~~~~~~~~~~~
# Distributed trace: 3 spans, both frontend.backend_call AND backend span
#   show status=ERROR.  This teaches "blast radius" — you can see exactly
#   how far the error propagated through the system.
# Both services' error_counter metrics increment.
# Both services log a WARNING/ERROR with the SAME trace_id.
# This is the clearest demonstration of metrics + logs + traces correlation.
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
@app.get("/backend-error", tags=["demo"])
async def backend_error():
    result = await _call_backend("/error")
    status_code = result["status_code"]
    return JSONResponse(
        status_code=status_code,
        content={
            "service": "frontend-api",
            "endpoint": "/backend-error",
            "status": "error",
            "message": "Proxied request to backend /error.",
            "downstream_result": result["body"],
        },
    )
