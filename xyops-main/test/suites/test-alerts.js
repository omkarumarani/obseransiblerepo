const assert = require('node:assert/strict');
const Tools = require('pixl-tools');

// helper: sleep
async function sleep(ms) {
	await new Promise(res => setTimeout(res, ms));
}

// helper: poll internal alerts until at least one is present
async function waitForActiveAlerts(ctx, opts = {}) {
	const timeout = opts.timeout || 20000;
	const interval = opts.interval || 250;
	const start = performance.now();
	
	while (performance.now() - start < timeout) {
		if (Tools.numKeys(ctx.xy.activeAlerts)) return;
		await sleep(interval);
	}
	
	throw new Error('Timed out waiting for active alerts');
}

// helper: wait for db query to return at least one row
async function waitForAlertsDatabase(ctx, opts = {}) {
	const timeout = opts.timeout || 20000;
	const interval = opts.interval || 250;
	const start = performance.now();
	
	while (performance.now() - start < timeout) {
		let { data } = await ctx.request.json(ctx.api_url + '/app/search_alerts/v1', { query: opts.query, offset: 0, limit: 1 });
		if (data.code !== 0) throw new Error('search_alerts failed');
		if (data.rows.length > 0) return data.rows;
		await sleep(interval);
	}
	
	throw new Error('Timed out waiting for alert db to show rows');
}

exports.tests = [
	
	async function test_api_get_alerts(test) {
		// get all alerts
		let { data } = await this.request.json( this.api_url + '/app/get_alerts/v1', {} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( data.rows.length, "expected rows" );
		assert.ok( Tools.findObject(data.rows, { id: 'load_avg_high' } ), "expected load_avg_high" );
		assert.ok( data.list.length, "expected list length" );
	},

	async function test_api_get_alert_missing_param(test) {
		// fetch alert with missing id param
		let { data } = await this.request.json( this.api_url + '/app/get_alert/v1', {} );
		assert.ok( !!data.code, "expected error for missing id" );
	},

	async function test_api_get_alert(test) {
		// fetch single alert by id
		let { data } = await this.request.json( this.api_url + '/app/get_alert/v1', { id: 'load_avg_high' } );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( data.alert, "expected alert in response" );
		assert.ok( data.alert.id == 'load_avg_high', "alert id == load_avg_high" );
	},

	async function test_api_get_alert_missing(test) {
		// fetch non-existent alert
		let { data } = await this.request.json( this.api_url + '/app/get_alert/v1', { id: 'nope' } );
		assert.ok( !!data.code, "expected error for missing alert" );
	},

	async function test_api_create_alert_missing_title(test) {
		// create alert without required title
		let { data } = await this.request.json( this.api_url + '/app/create_alert/v1', {
			"expression": "monitors.cpu_usage >= 90",
			"message": "CPU usage is too high: {{pct(monitors.cpu_usage)}}"
		});
		assert.ok( !!data.code, "expected error for missing title" );
	},

	async function test_api_create_alert_missing_expression(test) {
		// create alert without required expression
		let { data } = await this.request.json( this.api_url + '/app/create_alert/v1', {
			"title": "High CPU Usage",
			"message": "CPU usage is too high: {{pct(monitors.cpu_usage)}}"
		});
		assert.ok( !!data.code, "expected error for missing expression" );
	},
	
	async function test_api_create_alert(test) {
		// create new alert
		let { data } = await this.request.json( this.api_url + '/app/create_alert/v1', {
			"title": "High CPU Usage",
			"expression": "monitors.cpu_usage >= 90",
			"message": "CPU usage is too high: {{pct(monitors.cpu_usage)}}",
			"groups": [],
			"email": "",
			"web_hook": "",
			"monitor_id": "cpu_usage",
			"enabled": true,
			"samples": 1,
			"notes": ""
		});
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( data.alert, "expected alert in response" );
		assert.ok( data.alert.id, "expected alert.id in response" );
		
		// save our new alert id for later
		this.alert_id = data.alert.id;
	},
	
	async function test_api_get_new_alert(test) {
		// fetch our new alert by id
		let { data } = await this.request.json( this.api_url + '/app/get_alert/v1', { id: this.alert_id } );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( !!data.alert, "expected alert in response" );
		assert.ok( data.alert.id == this.alert_id, "alert id unexpected" );
	},
	
	async function test_api_update_alert(test) {
		// update our alert (shallow merge)
		let { data } = await this.request.json( this.api_url + '/app/update_alert/v1', {
			"id": this.alert_id,
			"notes": "unit test notes"
		});
		assert.ok( data.code === 0, "successful api response" );
	},

	async function test_api_update_alert_missing_id(test) {
		// update without id should error
		let { data } = await this.request.json( this.api_url + '/app/update_alert/v1', {
			"notes": "oops"
		});
		assert.ok( !!data.code, "expected error for missing id" );
	},
	
	async function test_api_get_updated_alert(test) {
		// make sure our changes took
		let { data } = await this.request.json( this.api_url + '/app/get_alert/v1', { id: this.alert_id } );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( !!data.alert, "expected alert in response" );
		assert.ok( data.alert.id == this.alert_id, "alert id unexpected" );
		assert.ok( data.alert.notes == "unit test notes", "unexpected alert notes" );
	},
	
	async function test_api_test_alert(test) {
		// test alert expression and message
		let { data } = await this.request.json( this.api_url + '/app/test_alert/v1', {
			"server": "satunit1",
			"expression": "monitors.cpu_usage >= 90",
			"message": "CPU usage is too high: {{pct(monitors.cpu_usage)}}",
		});
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( !data.result, "expected result to be false" );
		assert.ok( !!data.message, "expected message to be present" );
		assert.ok( !!data.message.match(/CPU usage is too high/), "unexpected message content" );
	},

	async function test_api_test_alert_missing_server(test) {
		// test alert missing server should error
		let { data } = await this.request.json( this.api_url + '/app/test_alert/v1', {
			"expression": "monitors.cpu_usage >= 90",
			"message": "CPU usage is too high: {{pct(monitors.cpu_usage)}}"
		});
		assert.ok( !!data.code, "expected error for missing server" );
	},

	async function test_api_test_alert_missing_expression(test) {
		// test alert missing expression should error
		let { data } = await this.request.json( this.api_url + '/app/test_alert/v1', {
			"server": "satunit1",
			"message": "CPU usage is too high: {{pct(monitors.cpu_usage)}}"
		});
		assert.ok( !!data.code, "expected error for missing expression" );
	},
	
	async function test_api_delete_alert(test) {
		// delete our alert
		let { data } = await this.request.json( this.api_url + '/app/delete_alert/v1', {
			"id": this.alert_id
		});
		assert.ok( data.code === 0, "successful api response" );
	},

	async function test_api_delete_alert_missing_id(test) {
		// delete without id should error
		let { data } = await this.request.json( this.api_url + '/app/delete_alert/v1', {} );
		assert.ok( !!data.code, "expected error for missing id" );
	},

	async function test_api_delete_alert_missing(test) {
		// delete non-existent alert
		let { data } = await this.request.json( this.api_url + '/app/delete_alert/v1', { id: 'nope' } );
		assert.ok( !!data.code, "expected error for missing alert" );
	},
	
	async function test_api_get_alert_deleted(test) {
		// make sure our deleted alert is no longer fetchable
		let { data } = await this.request.json( this.api_url + '/app/get_alert/v1', { 
			id: this.alert_id
		} );
		assert.ok( !!data.code, "expected error for missing alert" );
		delete this.alert_id;
	},
	
	async function test_api_create_alert_for_firing(test) {
		// create new alert that will always fire
		let { data } = await this.request.json( this.api_url + '/app/create_alert/v1', {
			"title": "Server Has Memory (LOL)",
			"expression": "memory.total > 0",
			"message": "Server has non-zero total memory: {{bytes(memory.total)}}",
			"groups": [],
			"email": "",
			"web_hook": "",
			"monitor_id": "",
			"enabled": true,
			"samples": 1,
			"notes": ""
		});
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( data.alert, "expected alert in response" );
		assert.ok( data.alert.id, "expected alert.id in response" );
		
		// save our new alert id for later
		this.alert_id = data.alert.id;
	},
	
	async function test_alert_fire(test) {
		// trigger an alert to fire (here be dragons)
		
		// first, clear out our server's last_time_code, so it doesn't block the duplicate monitoring submission
		let server = this.xy.servers['satunit1'];
		assert.ok( !!server, "Found our server object" );
		server.last_time_code = 0;
		
		// next, clear out the server hourly timeline, again so we don't get dupe-blocked
		var timeline_key = 'timeline/' + server.id + '/hourly';
		await new Promise(res => {
			this.xy.storage.listDelete( timeline_key, false, res );
		});
		
		// next, trigger our mock satellite to send in a minute monitoring data packet
		this.satellite.runMonitors({});
		
		// wait for alert to show up in memory
		await waitForActiveAlerts(this);
		
		assert.ok( Tools.numKeys(this.xy.activeAlerts) == 1, "only one alert is expected" );
		
		let alert = Object.values(this.xy.activeAlerts)[0];
		assert.ok( alert.alert == this.alert_id, "expected alert id" );
		
		// "exp": "memory.total > 0",
		assert.ok( !!alert.exp.match(/memory\.total/), "expected expression pattern in alert" );
		
		// "message": "Server has non-zero total memory: 906.2 MB",
		assert.ok( !!alert.message.match(/total\s+memory\:\s+(\d+)/), "expected message pattern in alert" );
		
		// now wait for alert to show up in db (async)
		let rows = await waitForAlertsDatabase(this, { query: '*' });
		
		assert.ok( rows.length > 0, "expected rows to be non-empty" );
		let db_alert = rows[0];
		
		assert.ok( db_alert.alert == this.alert_id, "expected alert id" );
		
		// "exp": "memory.total > 0",
		assert.ok( !!db_alert.exp.match(/memory\.total/), "expected expression pattern in alert" );
		
		// "message": "Server has non-zero total memory: 906.2 MB",
		assert.ok( !!db_alert.message.match(/total\s+memory\:\s+(\d+)/), "expected message pattern in alert" );
	}
	
];
