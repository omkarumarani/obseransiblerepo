const assert = require('node:assert/strict');

exports.tests = [
	
	async function test_get_user_activity(test) {
		// get user activity log
		let { data } = await this.request.json( this.api_url + '/app/get_user_activity', {
			offset: 0,
			limit: 50
		} );
		assert.ok( data.code === 0, "successful api response" );
		assert.ok( !!data.rows && Array.isArray(data.rows), "expected rows to be array" );
		assert.ok( data.rows.length > 0, "expected at least one row" );
	},
	
	async function test_user_settings(test) {
		// change some user settings
		let { data } = await this.request.json( this.api_url + '/app/user_settings', {
			"language": "en-US",
			"timezone": "America/Los_Angeles",
			"contrast": "high",
			"motion": "reduced"
		} );
		assert.ok( data.code === 0, "successful user settings update" );
		
		// fetch user using admin api to see our changes
		let resp = await this.request.json( this.api_url + '/user/admin_get_user', {
			"username": "testuser"
		} );
		assert.ok( resp.data.code === 0, "successful admin_get_user" );
		assert.ok( !!resp.data.user, "expected user object in response" );
		assert.ok( resp.data.user.contrast == "high", "expected high contrast in user object in response" );
	},
	
	async function test_user_logout_all(test) {
		// logout all sessions except the current one
		let { data } = await this.request.json( this.api_url + '/app/logout_all', {
			password: "testuser"
		} );
		assert.ok( data.code === 0, "successful api response" );
	}
	
];
