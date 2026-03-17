const assert = require('node:assert/strict');
const Tools = require('pixl-tools');

exports.tests = [

	async function test_api_get_groups(test) {
		// list all groups
		let { data } = await this.request.json( this.api_url + '/app/get_groups/v1', {} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( Array.isArray(data.rows), "expected rows array" );
		assert.ok( data.list && (data.list.length >= 0), "expected list metadata" );
	},

	async function test_api_get_group_missing_param(test) {
		// missing id param
		let { data } = await this.request.json( this.api_url + '/app/get_group/v1', {} );
		assert.ok( !!data.code, "expected error for missing id" );
	},

	async function test_api_get_group_missing(test) {
		// non-existent group
		let { data } = await this.request.json( this.api_url + '/app/get_group/v1', { id: 'nope' } );
		assert.ok( !!data.code, "expected error for missing group" );
	},

	async function test_api_create_group_missing_title(test) {
		// missing required title
		let { data } = await this.request.json( this.api_url + '/app/create_group/v1', {
			"hostname_match": ".+"
		});
		assert.ok( !!data.code, "expected error for missing title" );
	},

	async function test_api_create_group_missing_hostname(test) {
		// missing required hostname_match
		let { data } = await this.request.json( this.api_url + '/app/create_group/v1', {
			"title": "Unit Test Group"
		});
		assert.ok( !!data.code, "expected error for missing hostname_match" );
	},

	async function test_api_create_group_invalid_action(test) {
		// invalid alert action (invalid condition)
		let { data } = await this.request.json( this.api_url + '/app/create_group/v1', {
			"title": "Bad Group",
			"hostname_match": ".+",
			"alert_actions": [ { "enabled": true, "condition": "nope", "type": "email", "users": ["admin"] } ]
		});
		assert.ok( !!data.code, "expected error for invalid alert action" );
	},

	async function test_api_create_group(test) {
		// create new group
		let { data } = await this.request.json( this.api_url + '/app/create_group/v1', {
			"title": "Unit Test Group",
			"hostname_match": ".+",
			"notes": "Created by unit tests"
		});
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( data.group && data.group.id, "expected group in response" );
		this.group_id = data.group.id;
	},

	async function test_api_get_new_group(test) {
		// fetch our group
		let { data } = await this.request.json( this.api_url + '/app/get_group/v1', { id: this.group_id } );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( data.group && data.group.id === this.group_id, "group id unexpected" );
		assert.ok( data.group.title === 'Unit Test Group', "unexpected group title" );
		assert.ok( !!data.group.hostname_match, "expected hostname_match" );
	},

	async function test_api_update_group_missing_id(test) {
		// update without id should error
		let { data } = await this.request.json( this.api_url + '/app/update_group/v1', { title: 'oops' } );
		assert.ok( !!data.code, "expected error for missing id" );
	},

	async function test_api_update_group(test) {
		// update our group
		let { data } = await this.request.json( this.api_url + '/app/update_group/v1', {
			id: this.group_id,
			title: 'UTG v2',
			hostname_match: '^satunit'
		});
		assert.ok( data.code === 0, "successful api response" );
	},

	async function test_api_update_group_invalid_action(test) {
		// invalid alert action on update (missing users/email)
		let { data } = await this.request.json( this.api_url + '/app/update_group/v1', {
			id: this.group_id,
			alert_actions: [ { enabled: true, condition: 'error', type: 'email' } ]
		});
		assert.ok( !!data.code, "expected error for invalid alert action on update" );
	},

	async function test_api_get_updated_group(test) {
		// verify updates
		let { data } = await this.request.json( this.api_url + '/app/get_group/v1', { id: this.group_id } );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( data.group && data.group.title === 'UTG v2', "unexpected group title" );
		assert.ok( data.group.hostname_match === '^satunit', "unexpected hostname_match" );
	},

	async function test_api_delete_group_missing_id(test) {
		// delete without id should error
		let { data } = await this.request.json( this.api_url + '/app/delete_group/v1', {} );
		assert.ok( !!data.code, "expected error for missing id" );
	},

	async function test_api_delete_group_nonexistent(test) {
		// delete non-existent group should error
		let { data } = await this.request.json( this.api_url + '/app/delete_group/v1', { id: 'nope' } );
		assert.ok( !!data.code, "expected error for missing group" );
	},

	async function test_api_delete_group(test) {
		// delete our group
		let { data } = await this.request.json( this.api_url + '/app/delete_group/v1', { id: this.group_id } );
		assert.ok( data.code === 0, "successful api response" );
	},

	async function test_api_get_group_deleted(test) {
		// ensure deleted
		let { data } = await this.request.json( this.api_url + '/app/get_group/v1', { id: this.group_id } );
		assert.ok( !!data.code, "expected error for missing group" );
		delete this.group_id;
	},

	async function test_api_stub_multi_update_group(test) {
		// stubbed: skip multi_update_group
		assert.ok(true, 'stub multi_update_group');
	},

	async function test_api_stub_watch_group(test) {
		// stubbed: skip watch_group
		assert.ok(true, 'stub watch_group');
	},

	async function test_api_create_group_final(test) {
		// create a final group for other suites
		let { data } = await this.request.json( this.api_url + '/app/create_group/v1', {
			"title": "Unit Test Group Final",
			"hostname_match": ".+",
			"notes": "Keep me for future tests"
		});
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( data.group && data.group.id, "expected group in response" );
		this.group_final_id = data.group.id;
	},

	async function test_api_create_group_snapshot(test) {
		// create a snapshot for the final group and save the id
		let { data } = await this.request.json( this.api_url + '/app/create_group_snapshot/v1', {
			group: this.group_final_id
		});
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( data.id, "expected snapshot id in response" );
		this.group_snapshot_id = data.id;
	}

];
