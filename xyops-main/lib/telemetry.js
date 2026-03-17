// lib/telemetry.js
// ─────────────────────────────────────────────────────────────
// OpenTelemetry SDK bootstrap for xyOps
//
// Must be required BEFORE any other modules in main.js so that
// auto-instrumentation patches are in place before pixl-server
// initialises its HTTP server and WebSocket listeners.
//
// What this sets up:
//   1. TracerProvider  — every incoming HTTP request + WebSocket message
//      gets a SERVER span.  Job executions are wrapped in manual INTERNAL
//      spans (see lib/job.js integration notes below).
//   2. MeterProvider   — exports xyOps operational metrics to Prometheus
//      via the shared OTel collector:
//        xyops_jobs_started_total
//        xyops_jobs_completed_total
//        xyops_jobs_failed_total
//        xyops_alerts_fired_total
//        xyops_alerts_cleared_total
//   3. Auto-instrumentation:
//        @opentelemetry/instrumentation-http   — Node.js http/https module
//        @opentelemetry/instrumentation-express — Express-compatible servers
//        (pixl-server-web uses a custom HTTP layer on top of Node's http,
//         so http instrumentation captures the incoming requests)
//
// Environment variables consumed:
//   OTEL_SERVICE_NAME                 xyops  (default)
//   OTEL_EXPORTER_OTLP_ENDPOINT       http://otel-collector:4317
//   OTEL_EXPORTER_OTLP_PROTOCOL       grpc  (default)
//   OTEL_METRIC_EXPORT_INTERVAL_MS    15000
//   DEPLOYMENT_ENVIRONMENT            local
//
// How to use this module elsewhere in xyOps:
//   const { getTracer, meters } = require('./telemetry');
//   const tracer = getTracer();
//   const span = tracer.startSpan('my.operation');
//   // ... do work ...
//   span.end();
// ─────────────────────────────────────────────────────────────

'use strict';

const { NodeSDK }                        = require('@opentelemetry/sdk-node');
const { OTLPTraceExporter }              = require('@opentelemetry/exporter-trace-otlp-grpc');
const { OTLPMetricExporter }             = require('@opentelemetry/exporter-metrics-otlp-grpc');
const { PeriodicExportingMetricReader }  = require('@opentelemetry/sdk-metrics');
const { Resource }                       = require('@opentelemetry/resources');
const { getNodeAutoInstrumentations }    = require('@opentelemetry/auto-instrumentations-node');
const { trace, metrics }                 = require('@opentelemetry/api');
const {
  SEMRESATTRS_SERVICE_NAME,
  SEMRESATTRS_SERVICE_VERSION,
  SEMRESATTRS_DEPLOYMENT_ENVIRONMENT,
} = require('@opentelemetry/semantic-conventions');

// ── Configuration ──────────────────────────────────────────────────────────────
const OTEL_ENDPOINT   = process.env.OTEL_EXPORTER_OTLP_ENDPOINT || '';
const SERVICE_NAME    = process.env.OTEL_SERVICE_NAME            || 'xyops';
const SERVICE_VERSION = '1.0.28';
const ENVIRONMENT     = process.env.DEPLOYMENT_ENVIRONMENT       || 'local';

// ── Skip OTel setup when no collector is configured ────────────────────────────
// This prevents startup errors when running xyOps outside Docker
// (e.g. bare-metal dev mode) where no OTel collector is reachable.
if (!OTEL_ENDPOINT) {
	console.log('[OTel] OTEL_EXPORTER_OTLP_ENDPOINT not set — telemetry disabled (running without collector)');
	module.exports = {
		getTracer: () => trace.getTracer(SERVICE_NAME),
		getMeter:  () => metrics.getMeter(SERVICE_NAME),
		meters:    {},
	};
	return;
}

// ── Resource — identifies this service on every span and metric ─────────────────
const resource = new Resource({
	[SEMRESATTRS_SERVICE_NAME]:            SERVICE_NAME,
	[SEMRESATTRS_SERVICE_VERSION]:         SERVICE_VERSION,
	[SEMRESATTRS_DEPLOYMENT_ENVIRONMENT]:  ENVIRONMENT,
	'service.namespace': 'aiops',
});

// ── SDK initialisation ─────────────────────────────────────────────────────────
const sdk = new NodeSDK({
	resource,

	// Trace exporter: OTLP/gRPC to the shared collector
	traceExporter: new OTLPTraceExporter({
		// The SDK reads OTEL_EXPORTER_OTLP_ENDPOINT from env automatically.
		// Specifying it here explicitly ensures it is used even if the env
		// var format differs slightly (http:// vs grpc://).
		url: OTEL_ENDPOINT,
	}),

	// Metric exporter: push to collector every 15 s (aligns with Prometheus scrape)
	metricReader: new PeriodicExportingMetricReader({
		exporter: new OTLPMetricExporter({ url: OTEL_ENDPOINT }),
		exportIntervalMillis: parseInt(process.env.OTEL_METRIC_EXPORT_INTERVAL_MS || '15000', 10),
	}),

	// Auto-instrumentation bundle:
	//   - http / https                     → SERVER spans for incoming requests
	//   - DNS, net                         → span attributes on network calls
	//   - @opentelemetry/instrumentation-fs is disabled (too noisy in a file-heavy app)
	instrumentations: [
		getNodeAutoInstrumentations({
			'@opentelemetry/instrumentation-fs':       { enabled: false },
			'@opentelemetry/instrumentation-http':     { enabled: true  },
			'@opentelemetry/instrumentation-express':  { enabled: false }, // pixl uses own router
		}),
	],
});

// Start the SDK synchronously (must happen before any I/O)
sdk.start();
console.log(`[OTel] xyOps telemetry started  service=${SERVICE_NAME}  endpoint=${OTEL_ENDPOINT}`);

// ── Custom metric instruments ──────────────────────────────────────────────────
// These counters are pre-created here and exported as `meters` so that
// other xyOps lib files can import and use them without setting up their
// own meter.  Example in lib/job.js:
//
//   const { meters } = require('./telemetry');
//   meters.jobsStarted.add(1, { event_id: event.id, plugin: plugin.id });
//
const meter = metrics.getMeter(SERVICE_NAME);

const meters = {
	// Job lifecycle counters
	jobsStarted:   meter.createCounter('xyops_jobs_started_total',   { description: 'Total xyOps jobs started'    }),
	jobsCompleted: meter.createCounter('xyops_jobs_completed_total', { description: 'Total xyOps jobs completed'  }),
	jobsFailed:    meter.createCounter('xyops_jobs_failed_total',    { description: 'Total xyOps jobs failed'     }),

	// Job duration histogram (seconds)
	jobDuration: meter.createHistogram('xyops_job_duration_seconds', {
		description: 'xyOps job execution duration',
		unit:        's',
	}),

	// Alert lifecycle counters
	alertsFired:   meter.createCounter('xyops_alerts_fired_total',   { description: 'Total xyOps alerts fired'   }),
	alertsCleared: meter.createCounter('xyops_alerts_cleared_total', { description: 'Total xyOps alerts cleared' }),

	// Ticket counters
	ticketsCreated: meter.createCounter('xyops_tickets_created_total', { description: 'Total xyOps tickets created' }),
	ticketsClosed:  meter.createCounter('xyops_tickets_closed_total',  { description: 'Total xyOps tickets closed'  }),
};

// ── Graceful shutdown ──────────────────────────────────────────────────────────
// Flush in-flight spans and metrics before the process exits so the
// last few seconds of data are not lost.
process.on('SIGTERM', async () => {
	try {
		await sdk.shutdown();
		console.log('[OTel] Telemetry flushed and shut down.');
	} catch (err) {
		console.error('[OTel] Error during shutdown:', err);
	}
});

// ── Exports ────────────────────────────────────────────────────────────────────
module.exports = {
	/** Get a named Tracer for creating manual spans. */
	getTracer: (name) => trace.getTracer(name || SERVICE_NAME),
	/** Get the global Meter for creating additional instruments. */
	getMeter: (name) => metrics.getMeter(name || SERVICE_NAME),
	/** Pre-built metric instruments — import these in lib/job.js, lib/alert.js etc. */
	meters,
	/** The SDK instance — rarely needed directly. */
	sdk,
};
