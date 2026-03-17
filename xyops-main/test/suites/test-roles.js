const assert = require('node:assert/strict');
const Tools = require('pixl-tools');

exports.tests = [

    async function test_api_get_roles(test) {
        // list all roles
        let { data } = await this.request.json( this.api_url + '/app/get_roles/v1', {} );
        assert.ok( data.code === 0, 'successful api response' );
        assert.ok( Array.isArray(data.rows), 'expected rows array' );
        assert.ok( data.list && (data.list.length >= 0), 'expected list metadata' );
        // sanity check that built-in role exists
        assert.ok( Tools.findObject(data.rows, { id: 'all' } ), 'expected role "all" to exist' );
    },

    async function test_api_get_role_missing_param(test) {
        // missing id param
        let { data } = await this.request.json( this.api_url + '/app/get_role/v1', {} );
        assert.ok( !!data.code, 'expected error for missing id' );
    },

    async function test_api_get_role_missing(test) {
        // non-existent role
        let { data } = await this.request.json( this.api_url + '/app/get_role/v1', { id: 'nope' } );
        assert.ok( !!data.code, 'expected error for missing role' );
    },

    async function test_api_create_role_missing_title(test) {
        // missing required title
        let { data } = await this.request.json( this.api_url + '/app/create_role/v1', { enabled: true } );
        assert.ok( !!data.code, 'expected error for missing title' );
    },

    async function test_api_create_role(test) {
        // create new role
        let { data } = await this.request.json( this.api_url + '/app/create_role/v1', {
            title: 'Unit Test Role',
            enabled: true,
            icon: 'account-hard-hat',
            notes: 'Created by unit tests',
            categories: ['general'],
            groups: ['main'],
            privileges: { view_jobs: true }
        });
        assert.ok( data.code === 0, 'successful api response' );
        assert.ok( data.role && data.role.id, 'expected role in response' );
        this.role_id = data.role.id;
    },

    async function test_api_get_new_role(test) {
        // fetch our role
        let { data } = await this.request.json( this.api_url + '/app/get_role/v1', { id: this.role_id } );
        assert.ok( data.code === 0, 'successful api response' );
        assert.ok( data.role && data.role.id === this.role_id, 'role id unexpected' );
        assert.ok( data.role.title === 'Unit Test Role', 'unexpected role title' );
        assert.ok( data.role.privileges && (data.role.privileges.view_jobs === true), 'unexpected role privileges' );
    },

    async function test_api_update_role_missing_id(test) {
        // update without id should error
        let { data } = await this.request.json( this.api_url + '/app/update_role/v1', { title: 'oops' } );
        assert.ok( !!data.code, 'expected error for missing id' );
    },

    async function test_api_update_role(test) {
        // update our role
        let { data } = await this.request.json( this.api_url + '/app/update_role/v1', {
            id: this.role_id,
            title: 'UTR v2',
            enabled: false,
            categories: ['general']
        });
        assert.ok( data.code === 0, 'successful api response' );
    },

    async function test_api_get_updated_role(test) {
        // verify updates
        let { data } = await this.request.json( this.api_url + '/app/get_role/v1', { id: this.role_id } );
        assert.ok( data.code === 0, 'successful api response' );
        assert.ok( data.role && data.role.title === 'UTR v2', 'unexpected role title' );
        assert.ok( data.role.enabled === false, 'unexpected role enabled flag' );
        assert.ok( Array.isArray(data.role.categories) && data.role.categories[0] === 'general', 'unexpected categories content' );
    },

    async function test_api_delete_role_missing_id(test) {
        // delete without id should error
        let { data } = await this.request.json( this.api_url + '/app/delete_role/v1', {} );
        assert.ok( !!data.code, 'expected error for missing id' );
    },

    async function test_api_delete_role_nonexistent(test) {
        // delete non-existent role should error
        let { data } = await this.request.json( this.api_url + '/app/delete_role/v1', { id: 'nope' } );
        assert.ok( !!data.code, 'expected error for missing role' );
    },

    async function test_api_delete_role(test) {
        // delete our role
        let { data } = await this.request.json( this.api_url + '/app/delete_role/v1', { id: this.role_id } );
        assert.ok( data.code === 0, 'successful api response' );
    },

    async function test_api_get_role_deleted(test) {
        // ensure deleted
        let { data } = await this.request.json( this.api_url + '/app/get_role/v1', { id: this.role_id } );
        assert.ok( !!data.code, 'expected error for missing role' );
        delete this.role_id;
    }

];

