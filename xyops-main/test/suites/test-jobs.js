const assert = require('node:assert/strict');
const fs = require('fs');
const Tools = require('pixl-tools');

// helper: sleep
async function sleep(ms) {
	await new Promise(res => setTimeout(res, ms));
}

// helper: poll internal jobs until specified job id disappears
async function waitForJob(ctx, job_id, opts = {}) {
	const timeout = opts.timeout || 20000;
	const interval = opts.interval || 250;
	const start = performance.now();
	
	while (performance.now() - start < timeout) {
		let { data } = await ctx.request.json(ctx.api_url + '/app/get_active_jobs/v1', {});
		if (data.code !== 0) throw new Error('get_active_jobs failed');
		if (!data.rows.find(r => r.id === job_id)) return;
		await sleep(interval);
	}
	
	throw new Error('Timed out waiting for job to finish');
}

// helper: wait for all jobs, with optional criteria
async function waitForAllJobs(ctx, opts = {}) {
	const timeout = opts.timeout || 20000;
	const interval = opts.interval || 250;
	const criteria = opts.criteria || {}; // e.g. state:queued
	const start = performance.now();
	
	while (performance.now() - start < timeout) {
		let { data } = await ctx.request.json(ctx.api_url + '/app/get_active_jobs/v1', criteria);
		if (data.code !== 0) throw new Error('get_active_jobs failed');
		if (!data.rows.length) return;
		if (opts.max_jobs && (data.rows.length > opts.max_jobs)) throw new Error('max_jobs exceeded: ' + data.rows.length);
		
		// DEBUG: log all unique state counts
		var states = {};
		data.rows.forEach( function(job) {
			states[ job.state ] = (states[ job.state ] || 0) + 1;
		} );
		// console.log( "JOB STATES: " + JSON.stringify(states) );
		
		if (opts.max_active_jobs) {
			var active_jobs = Tools.findObjects( data.rows, { state: 'active' } );
			if (active_jobs.length > opts.max_active_jobs) throw new Error('max_jobs exceeded: ' + active_jobs.length);
		}
		
		await sleep(interval);
	}
	
	throw new Error('Timed out waiting for all jobs to finish');
}

exports.tests = [
	
	async function test_create_web_hook_for_job(test) {
		// create new web hook that points back to our echo API
		let { data } = await this.request.json( this.api_url + '/app/create_web_hook/v1', {
			title: 'Job Test Web Hook',
			enabled: true,
			url: this.api_url + '/app/echo?jobby=1',
			method: 'POST',
			headers: [
				{ name: 'Content-Type', value: 'application/json' },
				{ name: 'User-Agent', value: 'xyOps/WebHook' }
			],
			body: '{\n  "text": "{{text}}",\n  "content": "{{text}}"\n}',
			timeout: 10,
			retries: 0,
			follow: false,
			ssl_cert_bypass: false,
			max_per_day: 0,
			notes: 'Created by unit tests'
		});
		assert.ok( data.code === 0, 'successful api response' );
		assert.ok( data.web_hook && data.web_hook.id, 'expected web_hook in response' );
		this.web_hook_id = data.web_hook.id;
	},
	
	async function test_create_category_for_job(test) {
		// create a final category for use by job tests
		let { data } = await this.request.json( this.api_url + '/app/create_category/v1', {
			"title": "Job Test Category",
			"enabled": true,
			"color": "plain",
			"notes": "For job",
			"actions": [
				{ enabled: true, condition: 'start', type: 'web_hook', web_hook: this.web_hook_id }
			],
			"limits": []
		});
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( data.category && data.category.id, "expected category in response" );
		this.category_id = data.category.id;
	},
	
	async function test_create_event_for_job(test) {
		// create new event (non-workflow)
		let { data } = await this.request.json( this.api_url + '/app/create_event/v1', {
			"title": "Job Test Event",
			"enabled": true,
			"category": this.category_id, // inherit start action (web hook)
			"targets": ["main"],
			"algo": "random",
			"plugin": "shellplug",
			"params": { "script": "#!/bin/bash\necho hello\n", "annotate": false, "json": false },
			"limits": [ { enabled: true, type: 'time', duration: 60 } ],
			"actions": [ { enabled: true, condition: 'success', type: 'email', users: ['admin'] } ],
			"triggers": [ { "type": "manual", "enabled": true } ],
			"notes": "Created by unit tests"
		});
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( data.event && data.event.id, "expected event in response" );
		this.event_id = data.event.id;
	},
	
	async function test_create_secret_for_job(test) {
		// create a new secret with fields and assignments
		const fields = [
			{ name: 'DB_HOST', value: 'db.dev.internal' },
			{ name: 'DB_USER', value: 'appuser' },
			{ name: 'DB_PASS', value: 'CorrectHorseBatteryStaple' }
		];
		let { data } = await this.request.json( this.api_url + '/app/create_secret/v1', {
			title: 'Unit Test Secret',
			enabled: true,
			icon: '',
			notes: 'Created by unit tests',
			plugins: [],
			categories: [],
			events: [ this.event_id ],
			web_hooks: [],
			fields
		});
		assert.ok( data.code === 0, 'successful api response' );
		assert.ok( data.secret && data.secret.id, 'expected secret in response' );
		assert.ok( Array.isArray(data.secret.names) && data.secret.names.length === 3, 'expected names array derived from fields' );
		assert.ok( data.secret.names.includes('DB_PASS'), 'expected DB_PASS in names' );
		this.secret_id = data.secret.id;
	},

	async function test_create_tag_for_job(test) {
		// create a final tag for use by jobs
		let { data } = await this.request.json( this.api_url + '/app/create_tag/v1', {
			title: 'Job Test Tag',
			icon: 'tag',
			notes: 'Keep me for job tests'
		});
		assert.ok( data.code === 0, 'successful api response' );
		assert.ok( data.tag && data.tag.id, 'expected tag in response' );
		this.tag_id = data.tag.id;
	},
	
	async function test_run_job_basic(test) {
		// run basic job
		let { data } = await this.request.json( this.api_url + '/app/run_event/v1', {
			id: this.event_id,
			params: { duration: 1 },
			tags: [ this.tag_id ]
		});
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( data.id, "expected id in response" );
		this.job_id = data.id;
		
		// wait for job to complete
		await waitForJob( this, this.job_id );
	},
	
	async function test_get_job_basic(test) {
		// get completed job info
		let { data } = await this.request.json( this.api_url + '/app/get_job', {
			id: this.job_id
		});
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( data.job, "expected job in response" );
		
		let job = data.job;
		assert.ok( job.code == 0, "job was successful" );
		assert.ok( job.category == this.category_id, "job has correct category" );
		
		// actions
		assert.ok( !!job.actions, "job has actions array" );
		let actions = job.actions;
		assert.ok( !!Tools.findObject(actions, { type: 'email', code: 0 }), "job has successful email action" );
		assert.ok( !!Tools.findObject(actions, { type: 'web_hook', code: 0 }), "job has successful web hook action" );
		
		// tags
		assert.ok( !!job.tags, "job has tags array" );
		assert.ok( job.tags.includes(this.tag_id), "job tags has our tag" );
		
		// data + secrets
		assert.ok( !!job.data, "job has data object" );
		assert.ok( !!job.data.secrets, "job data has echoed secrets object" );
		assert.ok( job.data.secrets.DB_PASS == "CorrectHorseBatteryStaple", "correct secret in job data" );
	},
	
	async function test_create_simple_event_for_job(test) {
		// create simple event with no actions
		const category_id = this.category_final_id || 'general';
		let { data } = await this.request.json( this.api_url + '/app/create_event/v1', {
			"title": "Simple Test Event",
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
		this.simple_event_id = data.event.id;
	},
	
	async function test_run_jobs_many(test) {
		// run many jobs in parallel
		const self = this;
		const job_ids = [];
		const MAX_JOBS = 10;
		
		// create runner function
		const run_job = async function() {
			let { data } = await self.request.json( self.api_url + '/app/run_event/v1', {
				id: self.simple_event_id,
				params: { duration: 1 }
			});
			assert.ok( data.code === 0, "successful api response" );
			assert.ok( data.id, "expected id in response" );
			job_ids.push(data.id);
			
			// wait for job to complete
			await waitForJob( self, data.id );
		}; // run_job
		
		// run run_job 10 times in parallel, await all
		const runners = [];
		for (let i = 0; i < MAX_JOBS; i++) {
			runners.push(run_job());
		}
		
		// wait for all of them to finish
		await Promise.all(runners);
		
		// ensure all ids are unique
		assert.ok( job_ids.length == MAX_JOBS, "correct number of job ids" );
		assert.ok( new Set(job_ids).size === job_ids.length, "all job ids are unique" );
		
		// fetch all of them at once
		let { data } = await this.request.json( this.api_url + '/app/get_jobs', {
			ids: job_ids
		});
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( data.jobs, "expected jobs in response" );
		assert.ok( data.jobs.length == MAX_JOBS, "correct number of jobs in response" );
		assert.ok( Tools.findObjects(data.jobs, { complete: true, code: 0 }).length == MAX_JOBS, "all jobs are complete and successful" );
		
		job_ids.forEach( function(job_id, idx) {
			assert.ok( !!Tools.findObject(data.jobs, { id: job_id }), "found job idx " + idx );
		} );
	},
	
	async function test_update_simple_event_queue(test) {
		// update simple event to only allow 1 job at a time + queue
		const category_id = this.category_final_id || 'general';
		let { data } = await this.request.json( this.api_url + '/app/update_event/v1', {
			"id": this.simple_event_id,
			"limits": [
				{
					"type": "job",
					"enabled": true,
					"amount": 1
				},
				{
					"type": "queue",
					"enabled": true,
					"amount": 10
				}
			]
		});
		assert.ok( data.code === 0, "successful api response" );
	},
	
	async function test_run_jobs_queue(test) {
		// run many jobs using queue
		const self = this;
		const job_ids = [];
		const MAX_JOBS = 3;
		
		// create runner function
		const run_job = async function() {
			let { data } = await self.request.json( self.api_url + '/app/run_event/v1', {
				id: self.simple_event_id,
				params: { duration: 1 }
			});
			assert.ok( data.code === 0, "successful api response" );
			assert.ok( data.id, "expected id in response" );
			job_ids.push(data.id);
		}; // run_job
		
		// run run_job 10 times in parallel, await all
		const runners = [];
		for (let i = 0; i < MAX_JOBS; i++) {
			runners.push(run_job());
		}
		await Promise.all(runners);
		
		// jobs should all be queued now, with only 1 running at a time
		await waitForAllJobs( this, {
			max_active_jobs: 1
		} );
		
		// ensure all ids are unique
		assert.ok( job_ids.length == MAX_JOBS, "correct number of job ids" );
		assert.ok( new Set(job_ids).size === job_ids.length, "all job ids are unique" );
		
		// fetch all of them at once
		let { data } = await this.request.json( this.api_url + '/app/get_jobs', {
			ids: job_ids
		});
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( data.jobs, "expected jobs in response" );
		assert.ok( data.jobs.length == MAX_JOBS, "correct number of jobs in response" );
		assert.ok( Tools.findObjects(data.jobs, { complete: true, code: 0 }).length == MAX_JOBS, "all jobs are complete and successful" );
		
		job_ids.forEach( function(job_id, idx) {
			assert.ok( !!Tools.findObject(data.jobs, { id: job_id }), "found job idx " + idx );
		} );
	},
	
	async function test_update_simple_event_limit(test) {
		// update simple event to hit limit after N seconds
		const category_id = this.category_final_id || 'general';
		let { data } = await this.request.json( this.api_url + '/app/update_event/v1', {
			"id": this.simple_event_id,
			"limits": [
				{
					"type": "time",
					"enabled": true,
					"tags": [],
					"users": [],
					"email": "",
					"web_hook": this.web_hook_id,
					"text": "",
					"snapshot": false,
					"abort": false,
					"duration": 1
				}
			]
		});
		assert.ok( data.code === 0, "successful api response" );
	},
	
	async function test_run_job_limit_timeout(test) {
		// run job that will hit a max time limit and abort
		let { data } = await this.request.json( this.api_url + '/app/run_event/v1', {
			id: this.simple_event_id,
			params: { duration: 5 }
		});
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( data.id, "expected id in response" );
		let job_id = data.id;
		
		// wait for job to complete
		await waitForJob( this, job_id );
		
		// fetch job details
		let { data:jdata } = await this.request.json( this.api_url + '/app/get_job', { id: job_id });
		assert.ok( jdata.code === 0, "successful api response" );
		assert.ok( jdata.job, "expected job in response" );
		
		let job = jdata.job;
		assert.ok( job.code == 0, "job was successful" );
		
		assert.ok( !!job.limits, "found limits array in job" );
		assert.ok( !!job.limits.length == 1, "correct number of limits in job" );
		let limit = job.limits[0];
		
		assert.ok( limit.type == 'time', "expected time limit" );
		assert.ok( limit.code == 0, "expected limit code to be 0" );
	},
	
];
