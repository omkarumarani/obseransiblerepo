const assert = require('node:assert/strict');
const Tools = require('pixl-tools');

// helper: sleep
async function sleep(ms) {
	await new Promise(res => setTimeout(res, ms));
}

function testTriggers(ctx, triggers, epoch) {
	// test a set of triggers in our test event
	let self = ctx;
	let xy = self.xy;
	let event = Tools.findObject( xy.events, { id: self.schedule_event_id } );
	assert( !!event, "expected to find our event in xy.events" );
	
	// swap in our custom triggers
	let old_triggers = event.triggers;
	event.triggers = triggers;
	
	// hijack launchJob
	let jobs = [];
	let old_launchJob = xy.launchJob;
	xy.launchJob = function(job, callback) {
		self.logDebug(9, "Intercepted launchJob", job);
		jobs.push(job);
		if (callback) callback(null, job.id);
	};
	
	// run scheduler tick with custom time args (all happens in same thread)
	xy.schedulerMinuteTick({ epoch });
	
	// restore everything
	event.triggers = old_triggers;
	xy.launchJob = old_launchJob;
	
	// return array of launched jobs
	return jobs;
}

exports.tests = [
	
	async function test_create_event_for_scheduling(test) {
		// create new event (non-workflow)
		let { data } = await this.request.json( this.api_url + '/app/create_event/v1', {
			"title": "Job Test Event",
			"enabled": true,
			"category": 'general',
			"targets": ["main"],
			"algo": "random",
			"plugin": "shellplug",
			"params": { "script": "#!/bin/bash\necho hello\n", "annotate": false, "json": false },
			"limits": [  ],
			"actions": [  ],
			"triggers": [ { "type": "manual", "enabled": true } ],
			"notes": "Created by unit tests"
		});
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( data.event && data.event.id, "expected event in response" );
		this.schedule_event_id = data.event.id;
	},
	
	async function test_scheduler_negative(test) {
		// test epoch = 1766111340 (2025/12/18 18:29:00 Pacific)
		let epoch = 1766111340;
		let triggers = [
			{
				"enabled": true,
				"type": "schedule",
				"minutes": [ 28 ]
			}
		];
		let jobs = testTriggers(this, triggers, epoch);
		assert.ok( jobs.length == 0, "expected 0 jobs to launch" );
	},
	
	async function test_scheduler_every_minute(test) {
		// test epoch = 1766111340 (2025/12/18 18:29:00 Pacific)
		let epoch = 1766111340;
		let triggers = [
			{
				"enabled": true,
				"type": "schedule",
				"minutes": []
			}
		];
		let jobs = testTriggers(this, triggers, epoch);
		assert.ok( jobs.length == 1, "expected 1 job to launch" );
	},
	
	async function test_scheduler_hourly(test) {
		// test epoch = 1766111340 (2025/12/18 18:29:00 Pacific)
		let epoch = 1766111340;
		let triggers = [
			{
				"enabled": true,
				"type": "schedule",
				"minutes": [ 29 ]
			}
		];
		let jobs = testTriggers(this, triggers, epoch);
		assert.ok( jobs.length == 1, "expected 1 job to launch" );
	},
	
	async function test_scheduler_daily(test) {
		// test epoch = 1766111340 (2025/12/18 18:29:00 Pacific)
		let epoch = 1766111340;
		let triggers = [
			{
				"enabled": true,
				"type": "schedule",
				"hours": [ 18 ],
				"minutes": [ 29 ]
			}
		];
		let jobs = testTriggers(this, triggers, epoch);
		assert.ok( jobs.length == 1, "expected 1 job to launch" );
	},
	
	async function test_scheduler_weekly(test) {
		// test epoch = 1766111340 (2025/12/18 18:29:00 Pacific)
		let epoch = 1766111340;
		let triggers = [
			{
				"enabled": true,
				"type": "schedule",
				"weekdays": [ 4 ],
				"hours": [ 18 ],
				"minutes": [ 29 ]
			}
		];
		let jobs = testTriggers(this, triggers, epoch);
		assert.ok( jobs.length == 1, "expected 1 job to launch" );
	},
	
	async function test_scheduler_monthly(test) {
		// test epoch = 1766111340 (2025/12/18 18:29:00 Pacific)
		let epoch = 1766111340;
		let triggers = [
			{
				"enabled": true,
				"type": "schedule",
				"days": [ 18 ],
				"weekdays": [ 4 ],
				"hours": [ 18 ],
				"minutes": [ 29 ]
			}
		];
		let jobs = testTriggers(this, triggers, epoch);
		assert.ok( jobs.length == 1, "expected 1 job to launch" );
	},
	
	async function test_scheduler_yearly(test) {
		// test epoch = 1766111340 (2025/12/18 18:29:00 Pacific)
		let epoch = 1766111340;
		let triggers = [
			{
				"enabled": true,
				"type": "schedule",
				"months": [ 12 ],
				"days": [ 18 ],
				"weekdays": [ 4 ],
				"hours": [ 18 ],
				"minutes": [ 29 ]
			}
		];
		let jobs = testTriggers(this, triggers, epoch);
		assert.ok( jobs.length == 1, "expected 1 job to launch" );
	},
	
	async function test_scheduler_custom(test) {
		// test epoch = 1766111340 (2025/12/18 18:29:00 Pacific)
		let epoch = 1766111340;
		let triggers = [
			{
				"enabled": true,
				"type": "schedule",
				"years": [ 2025 ],
				"months": [ 12 ],
				"days": [ 18 ],
				"weekdays": [ 4 ],
				"hours": [ 18 ],
				"minutes": [ 29 ]
			}
		];
		let jobs = testTriggers(this, triggers, epoch);
		assert.ok( jobs.length == 1, "expected 1 job to launch" );
	},
	
	async function test_scheduler_custom_negative(test) {
		// test epoch = 1766111340 (2025/12/18 18:29:00 Pacific)
		let epoch = 1766111340;
		let triggers = [
			{
				"enabled": true,
				"type": "schedule",
				"years": [ 2025 ],
				"months": [ 12 ],
				"days": [ 18 ],
				"weekdays": [ 3 ],
				"hours": [ 18 ],
				"minutes": [ 29 ]
			}
		];
		let jobs = testTriggers(this, triggers, epoch);
		assert.ok( jobs.length == 0, "expected 0 jobs to launch" );
	},
	
	async function test_scheduler_timezone(test) {
		// test epoch = 1766111340 (2025/12/18 18:29:00 Pacific)
		let epoch = 1766111340;
		let triggers = [
			{
				"enabled": true,
				"type": "schedule",
				"hours": [ 21 ], // +3 hours from pacific
				"minutes": [ 29 ],
				"timezone": "America/New_York"
			}
		];
		let jobs = testTriggers(this, triggers, epoch);
		assert.ok( jobs.length == 1, "expected 1 job to launch" );
	},
	
	async function test_scheduler_timezone_negative(test) {
		// test epoch = 1766111340 (2025/12/18 18:29:00 Pacific)
		let epoch = 1766111340;
		let triggers = [
			{
				"enabled": true,
				"type": "schedule",
				"hours": [ 18 ], // wrong hour for ny
				"minutes": [ 29 ],
				"timezone": "America/New_York"
			}
		];
		let jobs = testTriggers(this, triggers, epoch);
		assert.ok( jobs.length == 0, "expected 0 jobs to launch" );
	},
	
	async function test_scheduler_interval(test) {
		// test epoch = 1766111340 (2025/12/18 18:29:00 Pacific)
		let epoch = 1766111340;
		let triggers = [
			{
				"enabled": true,
				"type": "interval",
				"duration": 90,
				"start": 1766111250
			}
		];
		let jobs = testTriggers(this, triggers, epoch);
		assert.ok( jobs.length == 1, "expected 1 job to launch" );
	},
	
	async function test_scheduler_interval_sub_minute(test) {
		// test epoch = 1766111340 (2025/12/18 18:29:00 Pacific)
		let epoch = 1766111340;
		let triggers = [
			{
				"enabled": true,
				"type": "interval",
				"duration": 30,
				"start": epoch
			}
		];
		let jobs = testTriggers(this, triggers, epoch);
		assert.ok( jobs.length == 2, "expected 2 jobs to launch" );
		
		// make sure jobs are spaced correctly
		assert.ok( jobs[0].state == 'start_delay', "expected job state to be start_delay" );
		assert.ok( jobs[0].until == epoch, "expected job until to be " + epoch );
		
		assert.ok( jobs[1].state == 'start_delay', "expected job state to be start_delay" );
		assert.ok( jobs[1].until == epoch + 30, "expected job until to be " + epoch + "+30" );
	},
	
	async function test_scheduler_single_shot(test) {
		// test epoch = 1766111340 (2025/12/18 18:29:00 Pacific)
		let epoch = 1766111340;
		let triggers = [
			{
				"enabled": true,
				"type": "single",
				"epoch": epoch
			}
		];
		let jobs = testTriggers(this, triggers, epoch);
		assert.ok( jobs.length == 1, "expected 1 job to launch" );
	},
	
	async function test_scheduler_catch_up(test) {
		// test epoch = 1766111340 (2025/12/18 18:29:00 Pacific)
		let epoch = 1766111340;
		let triggers = [
			{
				"enabled": true,
				"type": "schedule",
				"minutes": [ 27, 28, 29 ]
			},
			{
				"enabled": true,
				"type": "catchup"
			}
		];
		
		// set cursor state for event, set to 30 min in the past
		this.xy.putState( 'events/' + this.schedule_event_id + '/cursor', epoch - 1800 );
		
		let jobs = testTriggers(this, triggers, epoch);
		assert.ok( jobs.length == 3, "expected 3 jobs to launch" );
		
		// make sure jobs are set to the correct now times for catch-up
		assert.ok( jobs[0].now == epoch - 120, "expected job idx 0 to have now set to epoch-120" );
		assert.ok( jobs[1].now == epoch - 60, "expected job idx 1 to have now set to epoch-60" );
		assert.ok( jobs[2].now == epoch, "expected job idx 2 to have now set to epoch" );
	},
	
	async function test_scheduler_range_in(test) {
		// test epoch = 1766111340 (2025/12/18 18:29:00 Pacific)
		let epoch = 1766111340;
		let triggers = [
			{
				"enabled": true,
				"type": "schedule",
				"minutes": [ 29 ]
			},
			{
				"enabled": true,
				"type": "range",
				"start": epoch - 30,
				"end": epoch + 30
			}
		];
		let jobs = testTriggers(this, triggers, epoch);
		assert.ok( jobs.length == 1, "expected 1 job to launch" );
	},
	
	async function test_scheduler_range_out(test) {
		// test epoch = 1766111340 (2025/12/18 18:29:00 Pacific)
		let epoch = 1766111340;
		let triggers = [
			{
				"enabled": true,
				"type": "schedule",
				"minutes": [ 29 ]
			},
			{
				"enabled": true,
				"type": "range",
				"start": epoch + 1800,
				"end": epoch + 3600
			}
		];
		let jobs = testTriggers(this, triggers, epoch);
		assert.ok( jobs.length == 0, "expected 0 jobs to launch" );
	},
	
	async function test_scheduler_blackout_in(test) {
		// test epoch = 1766111340 (2025/12/18 18:29:00 Pacific)
		let epoch = 1766111340;
		let triggers = [
			{
				"enabled": true,
				"type": "schedule",
				"minutes": [ 29 ]
			},
			{
				"enabled": true,
				"type": "blackout",
				"start": epoch - 30,
				"end": epoch + 30
			}
		];
		let jobs = testTriggers(this, triggers, epoch);
		assert.ok( jobs.length == 0, "expected 0 jobs to launch" );
	},
	
	async function test_scheduler_blackout_out(test) {
		// test epoch = 1766111340 (2025/12/18 18:29:00 Pacific)
		let epoch = 1766111340;
		let triggers = [
			{
				"enabled": true,
				"type": "schedule",
				"minutes": [ 29 ]
			},
			{
				"enabled": true,
				"type": "blackout",
				"start": epoch + 1800,
				"end": epoch + 3600
			}
		];
		let jobs = testTriggers(this, triggers, epoch);
		assert.ok( jobs.length == 1, "expected 1 job to launch" );
	},
	
	async function test_scheduler_delay(test) {
		// test epoch = 1766111340 (2025/12/18 18:29:00 Pacific)
		let epoch = 1766111340;
		let triggers = [
			{
				"enabled": true,
				"type": "schedule",
				"minutes": [ 29 ]
			},
			{
				"enabled": true,
				"type": "delay",
				"duration": 30
			}
		];
		let jobs = testTriggers(this, triggers, epoch);
		assert.ok( jobs.length == 1, "expected 1 job to launch" );
		assert.ok( jobs[0].state == 'start_delay', "expected job state to be start_delay" );
		assert.ok( jobs[0].until == epoch + 30, "expected job until to be epoch+30" );
	},
	
	async function test_scheduler_precision(test) {
		// test epoch = 1766111340 (2025/12/18 18:29:00 Pacific)
		let epoch = 1766111340;
		let triggers = [
			{
				"enabled": true,
				"type": "schedule",
				"minutes": [ 29 ]
			},
			{
				"enabled": true,
				"type": "precision",
				"seconds": [ 0, 15, 30, 45 ]
			}
		];
		let jobs = testTriggers(this, triggers, epoch);
		assert.ok( jobs.length == 4, "expected 4 jobs to launch" );
		
		assert.ok( jobs[0].state == 'start_delay', "expected job idx 0 state to be start_delay" );
		assert.ok( jobs[0].until == epoch, "expected job idx 0 until to be epoch+0" );
		
		assert.ok( jobs[1].state == 'start_delay', "expected job idx 1 state to be start_delay" );
		assert.ok( jobs[1].until == epoch + 15, "expected job idx 1 until to be epoch+15" );
		
		assert.ok( jobs[2].state == 'start_delay', "expected job idx 2 state to be start_delay" );
		assert.ok( jobs[2].until == epoch + 30, "expected job idx 2 until to be epoch+30" );
		
		assert.ok( jobs[3].state == 'start_delay', "expected job idx 3 state to be start_delay" );
		assert.ok( jobs[3].until == epoch + 45, "expected job idx 3 until to be epoch+45" );
	},
	
	async function test_create_scheduler_plugin(test) {
		// create new scheduler plugin that always launches (+1 sec delay)
		let node_bin = Tools.findBinSync('node');
		assert.ok( !!node_bin, "expected to find node in PATH or standard dirs" );
		
		let { data } = await this.request.json( this.api_url + '/app/create_plugin/v1', {
			title: 'Unit Test Plugin',
			enabled: true,
			type: 'scheduler',
			command: node_bin,
			script: `setTimeout( function() { console.log( JSON.stringify({ "xy":1, "items": [ true ] }) ); }, 500 );\n`,
			params: [ { id: 'foo', type: 'text', title: 'Foo', value: '' } ],
			notes: 'Created by unit tests'
		});
		assert.ok( data.code === 0, 'successful api response' );
		assert.ok( data.plugin && data.plugin.id, 'expected plugin in response' );
		this.scheduler_plugin_id = data.plugin.id;
	},
	
	async function test_scheduler_plugin(test) {
		// test epoch = 1766111340 (2025/12/18 18:29:00 Pacific)
		let self = this;
		let epoch = 1766111340;
		let triggers = [
			{
				"enabled": true,
				"type": "schedule",
				"minutes": [ 29 ]
			},
			{
				"enabled": true,
				"type": "plugin",
				"plugin_id": this.scheduler_plugin_id,
				"params": {}
			}
		];
		let jobs = testTriggers(this, triggers, epoch);
		assert.ok( jobs.length == 0, "expected 0 jobs to launch immediately" );
		
		// job will launch deferred, so we have to wait for it
		let old_launchJob = this.xy.launchJob;
		let launched = [];
		this.xy.launchJob = function(job, callback) {
			self.logDebug(9, "Intercepted deferred launchJob", job);
			launched.push(job);
			if (callback) callback(null, job.id);
		};
		
		const timeout = 20000;
		const interval = 250;
		const start = performance.now();
		
		while (performance.now() - start < timeout) {
			if (launched.length) break;
			await sleep(interval);
		}
		
		assert.ok( launched.length == 1, "expected 1 job to launch deferred" );
		
		this.xy.launchJob = old_launchJob;
	},
	
];
