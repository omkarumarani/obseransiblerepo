#!/usr/bin/env node

// xyOps Server - Main entry point
// Copyright (c) 2019 - 2026 PixlCore LLC
// Released under the BSD 3-Clause License.
// See the LICENSE.md file in this repository.

// ── OpenTelemetry bootstrap ───────────────────────────────────────────────────
// Must be the VERY FIRST require so all auto-instrumentation patches are
// applied before pixl-server registers its HTTP listener and any other I/O.
require('./telemetry');
// ─────────────────────────────────────────────────────────────────────────────

// Check if Node.js version is old
if (process.version.match(/^v?(\d+)/) && (parseInt(RegExp.$1) < 16) && !process.env['XYOPS_OLD']) {
	console.error("\nERROR: You are using an incompatible version of Node.js (" + process.version + ").  Please upgrade to v16 or later.  Instructions: https://nodejs.org/en/download/package-manager\n\nTo ignore this error and run unsafely, set an XYOPS_OLD environment variable.  Do this at your own risk.\n");
	process.exit(1);
}

var Echo = require('./echo.js');

// chdir to the proper server root dir
process.chdir( require('path').dirname( __dirname ) );

// load pixl-server
var server = require('./loader.js');

server.on('init', function() {
	// setup fancy echo / repl system, if enabled
	Echo.setup(server);
});

server.startup( function() {
	// server startup complete
	process.title = server.__name + ' Server';
} );
