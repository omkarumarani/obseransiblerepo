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

// helper: load workflow template from fixtures
function getWorkflow(name) {
	const data = JSON.parse( fs.readFileSync('test/fixtures/workflows/' + name + '.json') );
	return data.items[0].data;
}

exports.tests = [
	
	async function test_workflow_decision(test) {
		// test multi-job workflow with decision controller
		let { data:wflow } = await this.request.json( this.api_url + '/app/create_event/v1', getWorkflow('decision'));
		assert.ok( wflow.code === 0, "successful api response" );
		assert.ok( wflow.event && wflow.event.id, "expected event in response" );
		let workflow_id = wflow.event.id;
		
		let { data } = await this.request.json( this.api_url + '/app/run_event/v1', { id: workflow_id });
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( data.id, "expected id in response" );
		let job_id = data.id;
		
		// wait for workflow to complete
		await waitForJob( this, job_id );
		
		// get job data
		let { data:jdata } = await this.request.json( this.api_url + '/app/get_job', { id: job_id });
		assert.ok( jdata.code === 0, "successful api response" );
		assert.ok( jdata.job, "expected job in response" );
		
		let job = jdata.job;
		let state = job.workflow.state;
		let jobs = job.workflow.jobs;
		
		// assertions here
		assert.ok( job.code == 0, "workflow completed successfully" );
		
		// job node IDs:
		// nk42jgcz = "Generate Data"
		// nqe6dmpv = "Hi" (data.num == 42)
		// n8qpiq1i = "There" (data.num != 42)
		
		assert.ok( jobs['nk42jgcz'], "Found jobs for node nk42jgcz");
		assert.ok( jobs['nk42jgcz'].length == 1, "Exactly one job for node nk42jgcz");
		
		assert.ok( jobs['nqe6dmpv'], "Found jobs for node nqe6dmpv");
		assert.ok( jobs['nqe6dmpv'].length == 1, "Exactly one job for node nqe6dmpv");
		
		var job_ids = [];
		for (var node_id in jobs) {
			jobs[node_id].forEach( function(job) { job_ids.push( job.id ); } );
		}
		assert.ok( job_ids.length == 2, "correct number of sub-jobs" );
		
		// fetch all of them at once
		let { data:sjdata } = await this.request.json( this.api_url + '/app/get_jobs', {
			ids: job_ids
		});
		assert.ok( sjdata.code === 0, "successful api response" );
		assert.ok( sjdata.jobs, "expected jobs in response" );
		assert.ok( sjdata.jobs.length == 2, "correct number of jobs in response" );
		assert.ok( Tools.findObjects(sjdata.jobs, { complete: true, code: 0 }).length == 2, "all jobs are complete and successful" );
		
		job_ids.forEach( function(job_id, idx) {
			assert.ok( !!Tools.findObject(sjdata.jobs, { id: job_id }), "found job idx " + idx );
		} );
	},
	
	async function test_workflow_repeat_join(test) {
		// test multi-job workflow with repeat and join controllers
		let { data:wflow } = await this.request.json( this.api_url + '/app/create_event/v1', getWorkflow('repeat-join'));
		assert.ok( wflow.code === 0, "successful api response" );
		assert.ok( wflow.event && wflow.event.id, "expected event in response" );
		let workflow_id = wflow.event.id;
		
		let { data } = await this.request.json( this.api_url + '/app/run_event/v1', { id: workflow_id });
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( data.id, "expected id in response" );
		let job_id = data.id;
		
		// wait for workflow to complete
		await waitForJob( this, job_id );
		
		// get job data
		let { data:jdata } = await this.request.json( this.api_url + '/app/get_job', { id: job_id });
		assert.ok( jdata.code === 0, "successful api response" );
		assert.ok( jdata.job, "expected job in response" );
		
		let job = jdata.job;
		let state = job.workflow.state;
		let jobs = job.workflow.jobs;
		
		// assertions here
		assert.ok( job.code == 0, "workflow completed successfully" );
		
		// job node IDs:
		// n5r4k75x = "Generate Data" (x5)
		// nbzpuk59 = "Finalize Data" (x1)
		
		assert.ok( jobs['n5r4k75x'], "Found jobs for node n5r4k75x");
		assert.ok( jobs['n5r4k75x'].length == 5, "Exactly five jobs for node n5r4k75x (generate data)");
		
		assert.ok( jobs['nbzpuk59'], "Found jobs for node nbzpuk59");
		assert.ok( jobs['nbzpuk59'].length == 1, "Exactly one job for node nbzpuk59 (finalize data)");
		
		var job_ids = [];
		for (var node_id in jobs) {
			jobs[node_id].forEach( function(job) { job_ids.push( job.id ); } );
		}
		assert.ok( job_ids.length == 6, "correct number of sub-jobs" );
		
		// fetch all of them at once
		let { data:sjdata } = await this.request.json( this.api_url + '/app/get_jobs', {
			ids: job_ids,
			verbose: true // need this for data!
		});
		assert.ok( sjdata.code === 0, "successful api response" );
		assert.ok( sjdata.jobs, "expected jobs in response" );
		assert.ok( sjdata.jobs.length == 6, "correct number of jobs in response" );
		assert.ok( Tools.findObjects(sjdata.jobs, { complete: true, code: 0 }).length == 6, "all jobs are complete and successful" );
		
		job_ids.forEach( function(job_id, idx) {
			assert.ok( !!Tools.findObject(sjdata.jobs, { id: job_id }), "found job idx " + idx );
		} );
		
		// make sure final job has correct merged (joined) data
		var final_job_id = jobs['nbzpuk59'][0].id;
		var final_job = Tools.findObject( sjdata.jobs, { id: final_job_id } );
		assert.ok( !!final_job, "Found final job in sea of completed jobs" );
		
		assert.ok( !!final_job.input, "Found input object in final job" );
		assert.ok( !!final_job.input.data, "Found input data in final job" );
		assert.ok( !!final_job.input.data.items, "Found items in input data in final job" );
		assert.ok( final_job.input.data.items.length == 5, "Correct number of items in input data in final job" );
		
		assert.ok( !!final_job.input.data.combined, "Found combined object in input data in final job" );
		assert.ok( final_job.input.data.combined.num == 42, "Correct data in combined object in input data in final job" );
	},
	
	async function test_workflow_discrete_join(test) {
		// test multi-job workflow with join controller
		let { data:wflow } = await this.request.json( this.api_url + '/app/create_event/v1', getWorkflow('discrete-join'));
		assert.ok( wflow.code === 0, "successful api response" );
		assert.ok( wflow.event && wflow.event.id, "expected event in response" );
		let workflow_id = wflow.event.id;
		
		let { data } = await this.request.json( this.api_url + '/app/run_event/v1', { id: workflow_id });
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( data.id, "expected id in response" );
		let job_id = data.id;
		
		// wait for workflow to complete
		await waitForJob( this, job_id );
		
		// get job data
		let { data:jdata } = await this.request.json( this.api_url + '/app/get_job', { id: job_id });
		assert.ok( jdata.code === 0, "successful api response" );
		assert.ok( jdata.job, "expected job in response" );
		
		// console.log( "\nWORKFLOW: ", JSON.stringify(jdata.job, null, "\t") );
		
		let job = jdata.job;
		let state = job.workflow.state;
		let jobs = job.workflow.jobs;
		
		// assertions here
		assert.ok( job.code == 0, "workflow completed successfully" );
		
		// job node IDs:
		// nxew2z8t = "Generate Data 1"
		// nriolmha = "Generate Data 2"
		// ndl4jafi = "Finalize Data"
		
		assert.ok( jobs['nxew2z8t'], "Found jobs for node nxew2z8t");
		assert.ok( jobs['nxew2z8t'].length == 1, "Exactly 1 job for node nxew2z8t (generate data 1)");
		
		assert.ok( jobs['nriolmha'], "Found jobs for node nriolmha");
		assert.ok( jobs['nriolmha'].length == 1, "Exactly 1 job for node nriolmha (generate data 1)");
		
		assert.ok( jobs['ndl4jafi'], "Found jobs for node ndl4jafi");
		assert.ok( jobs['ndl4jafi'].length == 1, "Exactly 1 job for node ndl4jafi (finalize data)");
		
		var job_ids = [];
		for (var node_id in jobs) {
			jobs[node_id].forEach( function(job) { job_ids.push( job.id ); } );
		}
		assert.ok( job_ids.length == 3, "correct number of sub-jobs ran" );
		
		// fetch all of them at once
		let { data:sjdata } = await this.request.json( this.api_url + '/app/get_jobs', {
			ids: job_ids,
			verbose: true // need this for data!
		});
		assert.ok( sjdata.code === 0, "successful api response" );
		assert.ok( sjdata.jobs, "expected jobs in response" );
		assert.ok( sjdata.jobs.length == 3, "correct number of jobs in response" );
		assert.ok( Tools.findObjects(sjdata.jobs, { complete: true, code: 0 }).length == 3, "all jobs are complete and successful" );
		
		job_ids.forEach( function(job_id, idx) {
			assert.ok( !!Tools.findObject(sjdata.jobs, { id: job_id }), "found job idx " + idx );
		} );
		
		// make sure final job has correct merged (joined) data
		var final_job_id = jobs['ndl4jafi'][0].id;
		var final_job = Tools.findObject( sjdata.jobs, { id: final_job_id } );
		assert.ok( !!final_job, "Found final job in sea of completed jobs" );
		
		// console.log( "\nFINAL JOB: ", JSON.stringify(final_job, null, "\t") );
		
		assert.ok( !!final_job.input, "Found input object in final job" );
		assert.ok( !!final_job.input.data, "Found input data in final job" );
		assert.ok( !!final_job.input.data.items, "Found items in input data in final job" );
		assert.ok( final_job.input.data.items.length == 2, "Correct number of items in input data in final job" );
		
		assert.ok( !!final_job.input.data.combined, "Found combined object in input data in final job" );
		assert.ok( final_job.input.data.combined.num == 42, "Correct data in combined object in input data in final job" );
	},
	
	async function test_workflow_multiplex(test) {
		// test multi-job workflow with multiplex controller
		let { data:wflow } = await this.request.json( this.api_url + '/app/create_event/v1', getWorkflow('multiplex'));
		assert.ok( wflow.code === 0, "successful api response" );
		assert.ok( wflow.event && wflow.event.id, "expected event in response" );
		let workflow_id = wflow.event.id;
		
		let { data } = await this.request.json( this.api_url + '/app/run_event/v1', { id: workflow_id });
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( data.id, "expected id in response" );
		let job_id = data.id;
		
		// wait for workflow to complete
		await waitForJob( this, job_id );
		
		// get job data
		let { data:jdata } = await this.request.json( this.api_url + '/app/get_job', { id: job_id });
		assert.ok( jdata.code === 0, "successful api response" );
		assert.ok( jdata.job, "expected job in response" );
		
		let job = jdata.job;
		let state = job.workflow.state;
		let jobs = job.workflow.jobs;
		
		// console.log( "\nWORKFLOW: ", JSON.stringify(jdata.job, null, "\t") );
		
		// assertions here
		assert.ok( job.code == 0, "workflow completed successfully" );
		
		// job node IDs:
		// nizewm42 = job
		// n68r2e20 = action
		
		assert.ok( jobs['nizewm42'], "Found jobs for node nizewm42");
		assert.ok( jobs['nizewm42'].length == 1, "Exactly one job for node nizewm42");
		
		var job_ids = [];
		for (var node_id in jobs) {
			jobs[node_id].forEach( function(job) { job_ids.push( job.id ); } );
		}
		assert.ok( job_ids.length == 1, "correct number of sub-jobs" );
		
		// fetch all of them at once
		let { data:sjdata } = await this.request.json( this.api_url + '/app/get_jobs', {
			ids: job_ids
		});
		assert.ok( sjdata.code === 0, "successful api response" );
		assert.ok( sjdata.jobs, "expected jobs in response" );
		assert.ok( sjdata.jobs.length == 1, "correct number of jobs in response" );
		assert.ok( Tools.findObjects(sjdata.jobs, { complete: true, code: 0 }).length == 1, "all jobs are complete and successful" );
		
		job_ids.forEach( function(job_id, idx) {
			assert.ok( !!Tools.findObject(sjdata.jobs, { id: job_id }), "found job idx " + idx );
		} );
		
		// make sure action ran
		assert.ok( !!state['n68r2e20'], "found state for action node" );
		assert.ok( state['n68r2e20'].code == 0, "action node has code 0" );
		assert.ok( !!state['n68r2e20'].date, "action node has date prop" );
	},
	
	async function test_workflow_split(test) {
		// test multi-job workflow with split controller
		let { data:wflow } = await this.request.json( this.api_url + '/app/create_event/v1', getWorkflow('split'));
		assert.ok( wflow.code === 0, "successful api response" );
		assert.ok( wflow.event && wflow.event.id, "expected event in response" );
		let workflow_id = wflow.event.id;
		
		let { data } = await this.request.json( this.api_url + '/app/run_event/v1', { 
			id: workflow_id,
			input: {
				data: {
					list: ["Item A", "Item B", "Item C"]
				}
			}
		});
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( data.id, "expected id in response" );
		let job_id = data.id;
		
		// wait for workflow to complete
		await waitForJob( this, job_id );
		
		// get job data
		let { data:jdata } = await this.request.json( this.api_url + '/app/get_job', { id: job_id });
		assert.ok( jdata.code === 0, "successful api response" );
		assert.ok( jdata.job, "expected job in response" );
		
		let job = jdata.job;
		let state = job.workflow.state;
		let jobs = job.workflow.jobs;
		
		// console.log( "\nWORKFLOW: ", JSON.stringify(jdata.job, null, "\t") );
		
		// assertions here
		assert.ok( job.code == 0, "workflow completed successfully" );
		
		// job node IDs:
		// nan42bp0 = job
		
		assert.ok( jobs['nan42bp0'], "Found jobs for node nan42bp0");
		assert.ok( jobs['nan42bp0'].length == 3, "Exactly three jobs for node nan42bp0");
		
		var job_ids = [];
		for (var node_id in jobs) {
			jobs[node_id].forEach( function(job) { job_ids.push( job.id ); } );
		}
		assert.ok( job_ids.length == 3, "correct number of sub-jobs" );
		
		// fetch all of them at once
		let { data:sjdata } = await this.request.json( this.api_url + '/app/get_jobs', {
			ids: job_ids,
			verbose: true
		});
		assert.ok( sjdata.code === 0, "successful api response" );
		assert.ok( sjdata.jobs, "expected jobs in response" );
		assert.ok( sjdata.jobs.length == 3, "correct number of jobs in response" );
		assert.ok( Tools.findObjects(sjdata.jobs, { complete: true, code: 0 }).length == 3, "all jobs are complete and successful" );
		
		// make sure each sub-job was assigned one item from our split
		var split_items = [];
		job_ids.forEach( function(job_id, idx) {
			var sub_job = Tools.findObject(sjdata.jobs, { id: job_id });
			// console.log( "\nJOB " + idx + ": ", JSON.stringify(sub_job, null, "\t") );
			assert.ok( !!sub_job, "found sub-job idx " + idx );
			assert.ok( !!sub_job.input, `found sub-job idx ${idx} input object` );
			assert.ok( !!sub_job.input.data, `found sub-job idx ${idx} input.data object` );
			assert.ok( !!sub_job.input.data.item, `found sub-job idx ${idx} input.data.item` );
			assert.ok( !!sub_job.input.data.item.match(/^Item\s+\w+$/), `sub-job idx ${idx} input.data.item matches expected string pattern` );
			split_items.push( sub_job.input.data.item );
		} );
		
		// make sure all three items are unique
		assert.ok( new Set(split_items).size === split_items.length, "all split item strings are unique" );
	}

];
