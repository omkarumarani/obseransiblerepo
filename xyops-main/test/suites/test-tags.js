const assert = require('node:assert/strict');
const Tools = require('pixl-tools');

exports.tests = [

	async function test_api_get_tags(test) {
		// list all tags
		let { data } = await this.request.json( this.api_url + '/app/get_tags/v1', {} );
		assert.ok( data.code === 0, 'successful api response' );
		assert.ok( Array.isArray(data.rows), 'expected rows array' );
		assert.ok( data.list && (data.list.length >= 0), 'expected list metadata' );
	},

	async function test_api_get_tag_missing_param(test) {
		// missing id param
		let { data } = await this.request.json( this.api_url + '/app/get_tag/v1', {} );
		assert.ok( !!data.code, 'expected error for missing id' );
	},

	async function test_api_get_tag_missing(test) {
		// non-existent tag
		let { data } = await this.request.json( this.api_url + '/app/get_tag/v1', { id: 'nope' } );
		assert.ok( !!data.code, 'expected error for missing tag' );
	},

	async function test_api_create_tag_missing_title(test) {
		// missing required title
		let { data } = await this.request.json( this.api_url + '/app/create_tag/v1', { icon: 'alert-rhombus' } );
		assert.ok( !!data.code, 'expected error for missing title' );
	},

	async function test_api_create_tag_invalid_id(test) {
		// invalid id (hyphens not allowed by /^[-\w]+$/ expectation is only \w and first alnum)
		let { data } = await this.request.json( this.api_url + '/app/create_tag/v1', {
			id: 'bad-id',
			title: 'Bad Tag'
		});
		assert.ok( !!data.code, 'expected error for invalid id' );
	},

	async function test_api_create_tag(test) {
		// create new tag
		let { data } = await this.request.json( this.api_url + '/app/create_tag/v1', {
			title: 'Unit Test Tag',
			icon: 'alert-rhombus',
			notes: 'Created by unit tests'
		});
		assert.ok( data.code === 0, 'successful api response' );
		assert.ok( data.tag && data.tag.id, 'expected tag in response' );
		this.tag_id = data.tag.id;
	},

	async function test_api_get_new_tag(test) {
		// fetch our tag
		let { data } = await this.request.json( this.api_url + '/app/get_tag/v1', { id: this.tag_id } );
		assert.ok( data.code === 0, 'successful api response' );
		assert.ok( data.tag && data.tag.id === this.tag_id, 'tag id unexpected' );
		assert.ok( data.tag.title === 'Unit Test Tag', 'unexpected tag title' );
	},

	async function test_api_update_tag_missing_id(test) {
		// update without id should error
		let { data } = await this.request.json( this.api_url + '/app/update_tag/v1', { title: 'oops' } );
		assert.ok( !!data.code, 'expected error for missing id' );
	},

	async function test_api_update_tag(test) {
		// update our tag
		let { data } = await this.request.json( this.api_url + '/app/update_tag/v1', {
			id: this.tag_id,
			title: 'UTT v2',
			icon: 'star',
			notes: 'unit test notes'
		});
		assert.ok( data.code === 0, 'successful api response' );
	},

	async function test_api_get_updated_tag(test) {
		// verify updates
		let { data } = await this.request.json( this.api_url + '/app/get_tag/v1', { id: this.tag_id } );
		assert.ok( data.code === 0, 'successful api response' );
		assert.ok( data.tag && data.tag.title === 'UTT v2', 'unexpected tag title' );
		assert.ok( data.tag.icon === 'star', 'unexpected tag icon' );
		assert.ok( data.tag.notes === 'unit test notes', 'unexpected tag notes' );
	},

	async function test_api_delete_tag_missing_id(test) {
		// delete without id should error
		let { data } = await this.request.json( this.api_url + '/app/delete_tag/v1', {} );
		assert.ok( !!data.code, 'expected error for missing id' );
	},

	async function test_api_delete_tag_nonexistent(test) {
		// delete non-existent tag should error
		let { data } = await this.request.json( this.api_url + '/app/delete_tag/v1', { id: 'nope' } );
		assert.ok( !!data.code, 'expected error for missing tag' );
	},

	async function test_api_delete_tag(test) {
		// delete our tag
		let { data } = await this.request.json( this.api_url + '/app/delete_tag/v1', { id: this.tag_id } );
		assert.ok( data.code === 0, 'successful api response' );
	},

	async function test_api_get_tag_deleted(test) {
		// ensure deleted
		let { data } = await this.request.json( this.api_url + '/app/get_tag/v1', { id: this.tag_id } );
		assert.ok( !!data.code, 'expected error for missing tag' );
		delete this.tag_id;
	}

];

