const assert = require('node:assert/strict');
const Tools = require('pixl-tools');

exports.tests = [
	
	// Jobs
	
	async function test_api_search_jobs_basic(test) {
		// job search
		let { data } = await this.request.json( this.api_url + '/app/search_jobs/v1', {
			query: '*',
			offset: 0,
			limit: 25
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length > 0, "expected rows to be populated" );
		assert.ok( data.list && (data.list.length >= 0), "expected list metadata" );
	},
	
	async function test_api_search_jobs_pagination(test) {
		// job search (page 2)
		let { data } = await this.request.json( this.api_url + '/app/search_jobs/v1', {
			query: '*',
			offset: 25,
			limit: 25
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length > 0, "expected rows to be populated" );
		assert.ok( data.list && (data.list.length > 25), "expected list length to be greater than limit" );
	},
	
	async function test_api_search_jobs_event(test) {
		// job search for specific event
		let { data } = await this.request.json( this.api_url + '/app/search_jobs/v1', {
			query: 'event:' + this.event_id,
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length > 0, "expected rows to be populated" );
		assert.ok( data.list && (data.list.length >= 0), "expected list metadata" );
	},
	
	async function test_api_search_jobs_category(test) {
		// job search for specific category
		let { data } = await this.request.json( this.api_url + '/app/search_jobs/v1', {
			query: 'category:' + this.category_id,
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length > 0, "expected rows to be populated" );
		assert.ok( data.list && (data.list.length >= 0), "expected list metadata" );
	},
	
	async function test_api_search_jobs_tag(test) {
		// job search for specific tag
		let { data } = await this.request.json( this.api_url + '/app/search_jobs/v1', {
			query: 'tags:' + this.tag_id,
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length > 0, "expected rows to be populated" );
		assert.ok( data.list && (data.list.length >= 0), "expected list metadata" );
	},
	
	async function test_api_search_jobs_source(test) {
		// job search for specific source
		let { data } = await this.request.json( this.api_url + '/app/search_jobs/v1', {
			query: 'source:workflow',
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length > 0, "expected rows to be populated" );
		assert.ok( data.list && (data.list.length >= 0), "expected list metadata" );
	},
	
	async function test_api_search_jobs_date_range(test) {
		// job search for date range
		let { data } = await this.request.json( this.api_url + '/app/search_jobs/v1', {
			query: 'date:<=now',
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length > 0, "expected rows to be populated" );
		assert.ok( data.list && (data.list.length >= 0), "expected list metadata" );
	},
	
	async function test_api_search_jobs_negative(test) {
		// job search for something that should not match
		let { data } = await this.request.json( this.api_url + '/app/search_jobs/v1', {
			query: 'tags:sdfkjhsdkfhdskjf',
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length == 0, "expected rows to be empty" );
		assert.ok( data.list && (data.list.length == 0), "expected list metadata" );
	},
	
	// Servers
	
	async function test_api_search_servers_basic(test) {
		// server search
		let { data } = await this.request.json( this.api_url + '/app/search_servers/v1', {
			query: '*',
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length > 0, "expected rows to be populated" );
		assert.ok( data.list && (data.list.length >= 0), "expected list metadata" );
		
		var server = data.rows[0];
		assert.ok( !!server, "expected server in rows idx 0" );
		assert.ok( server.id == 'satunit1', "expected correct server id" );
		assert.ok( !!server.info, "expected server to have info object" );
		assert.ok( server.info.platform == 'linux', "expected server.info.platform to be linux" );
	},
	
	async function test_api_search_servers_keywords(test) {
		// server search for specific keyword
		let { data } = await this.request.json( this.api_url + '/app/search_servers/v1', {
			query: 'keywords:linux',
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length > 0, "expected rows to be populated" );
		assert.ok( data.list && (data.list.length >= 0), "expected list metadata" );
		
		var server = data.rows[0];
		assert.ok( !!server, "expected server in rows idx 0" );
		assert.ok( server.id == 'satunit1', "expected correct server id" );
	},
	
	async function test_api_search_servers_groups(test) {
		// server search for specific group
		let { data } = await this.request.json( this.api_url + '/app/search_servers/v1', {
			query: 'groups:' + this.group_final_id,
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length > 0, "expected rows to be populated" );
		assert.ok( data.list && (data.list.length >= 0), "expected list metadata" );
		
		var server = data.rows[0];
		assert.ok( !!server, "expected server in rows idx 0" );
		assert.ok( server.id == 'satunit1', "expected correct server id" );
	},
	
	async function test_api_search_servers_created_range(test) {
		// server search for created range
		let { data } = await this.request.json( this.api_url + '/app/search_servers/v1', {
			query: 'created:<=now',
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length > 0, "expected rows to be populated" );
		assert.ok( data.list && (data.list.length >= 0), "expected list metadata" );
		
		var server = data.rows[0];
		assert.ok( !!server, "expected server in rows idx 0" );
		assert.ok( server.id == 'satunit1', "expected correct server id" );
	},
	
	async function test_api_search_servers_negative(test) {
		// server search for something that should not match
		let { data } = await this.request.json( this.api_url + '/app/search_servers/v1', {
			query: 'keywords:gdljhflskdhfl1234',
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length == 0, "expected rows to be empty" );
		assert.ok( data.list && (data.list.length == 0), "expected list metadata" );
	},
	
	// Alerts
	
	async function test_api_search_alerts_basic(test) {
		// alert search
		let { data } = await this.request.json( this.api_url + '/app/search_alerts/v1', {
			query: '*',
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length > 0, "expected rows to be populated" );
		assert.ok( data.list && (data.list.length >= 0), "expected list metadata" );
		
		var alert = data.rows[0];
		assert.ok( !!alert, "expected alert in rows idx 0" );
		assert.ok( alert.alert == this.alert_id, "expected correct alert id" );
	},
	
	async function test_api_search_alerts_def(test) {
		// alert search by definition
		let { data } = await this.request.json( this.api_url + '/app/search_alerts/v1', {
			query: 'alert:' + this.alert_id,
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length > 0, "expected rows to be populated" );
		assert.ok( data.list && (data.list.length >= 0), "expected list metadata" );
		
		var alert = data.rows[0];
		assert.ok( !!alert, "expected alert in rows idx 0" );
		assert.ok( alert.alert == this.alert_id, "expected correct alert id" );
	},
	
	async function test_api_search_alerts_active(test) {
		// alert search by active flag
		let { data } = await this.request.json( this.api_url + '/app/search_alerts/v1', {
			query: 'active:true',
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length > 0, "expected rows to be populated" );
		assert.ok( data.list && (data.list.length >= 0), "expected list metadata" );
		
		var alert = data.rows[0];
		assert.ok( !!alert, "expected alert in rows idx 0" );
		assert.ok( alert.alert == this.alert_id, "expected correct alert id" );
	},
	
	async function test_api_search_alerts_start_range(test) {
		// alert search for start range
		let { data } = await this.request.json( this.api_url + '/app/search_alerts/v1', {
			query: 'start:<=now',
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length > 0, "expected rows to be populated" );
		assert.ok( data.list && (data.list.length >= 0), "expected list metadata" );
		
		var alert = data.rows[0];
		assert.ok( !!alert, "expected alert in rows idx 0" );
		assert.ok( alert.alert == this.alert_id, "expected correct alert id" );
	},
	
	async function test_api_search_alerts_negative(test) {
		// alert search for something that should not match
		let { data } = await this.request.json( this.api_url + '/app/search_alerts/v1', {
			query: 'alert:gdljhflskdhfl1234',
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length == 0, "expected rows to be empty" );
		assert.ok( data.list && (data.list.length == 0), "expected list metadata" );
	},
	
	// Snapshots
	
	async function test_api_search_snapshots_basic(test) {
		// snapshot search
		let { data } = await this.request.json( this.api_url + '/app/search_snapshots/v1', {
			query: '*',
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length > 0, "expected rows to be populated" );
		assert.ok( data.list && (data.list.length >= 0), "expected list metadata" );
	},
	
	async function test_api_search_snapshots_type_source(test) {
		// snapshot search for specific type and source
		let { data } = await this.request.json( this.api_url + '/app/search_snapshots/v1', {
			query: 'type:server source:user',
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length > 0, "expected rows to be populated" );
		assert.ok( data.list && (data.list.length >= 0), "expected list metadata" );
		
		var snap = data.rows[0];
		assert.ok( !!snap, "expected snap in rows idx 0" );
		assert.ok( snap.id == this.final_snapshot_id, "expected correct snap id" );
	},
	
	async function test_api_search_snapshots_date_range(test) {
		// snapshot search for date range
		let { data } = await this.request.json( this.api_url + '/app/search_snapshots/v1', {
			query: 'date:<=now',
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length > 0, "expected rows to be populated" );
		assert.ok( data.list && (data.list.length >= 0), "expected list metadata" );
	},
	
	async function test_api_search_snapshots_negative(test) {
		// snapshot search for criteria that should not match
		let { data } = await this.request.json( this.api_url + '/app/search_snapshots/v1', {
			query: 'server:adkljfhdsklfh',
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length == 0, "expected rows to be empty" );
		assert.ok( data.list && (data.list.length == 0), "expected list metadata" );
	},
	
	// Activity
	
	async function test_api_search_activity_basic(test) {
		// activity search
		let { data } = await this.request.json( this.api_url + '/app/search_activity/v1', {
			query: '*',
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length > 0, "expected rows to be populated" );
		assert.ok( data.list && (data.list.length >= 0), "expected list metadata" );
	},
	
	async function test_api_search_activity_action(test) {
		// activity search for specific action
		let { data } = await this.request.json( this.api_url + '/app/search_activity/v1', {
			query: 'action:event_create',
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length > 0, "expected rows to be populated" );
		assert.ok( data.list && (data.list.length >= 0), "expected list metadata" );
	},
	
	async function test_api_search_activity_keywords(test) {
		// activity search for specific keywords
		let { data } = await this.request.json( this.api_url + '/app/search_activity/v1', {
			query: 'keywords:testuser',
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length > 0, "expected rows to be populated" );
		assert.ok( data.list && (data.list.length >= 0), "expected list metadata" );
	},
	
	async function test_api_search_activity_date_range(test) {
		// activity search for date range
		let { data } = await this.request.json( this.api_url + '/app/search_activity/v1', {
			query: 'date:<=now',
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length > 0, "expected rows to be populated" );
		assert.ok( data.list && (data.list.length >= 0), "expected list metadata" );
	},
	
	async function test_api_search_activity_negative(test) {
		// activity search for criteria that should not match
		let { data } = await this.request.json( this.api_url + '/app/search_activity/v1', {
			query: 'keywords:adkljfhdsklfh',
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length == 0, "expected rows to be empty" );
		assert.ok( data.list && (data.list.length == 0), "expected list metadata" );
	},
	
	// Revision History
	
	async function test_api_search_revision_history(test) {
		// revision history search
		let { data } = await this.request.json( this.api_url + '/app/search_revision_history/v1', {
			type: 'events',
			query: '',
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.rows.length > 0, "expected rows to be populated" );
		assert.ok( data.list && (data.list.length >= 0), "expected list metadata" );
	}
	
];
