const assert = require('node:assert/strict');
const Tools = require('pixl-tools');

exports.tests = [

    async function test_api_get_plugins(test) {
        // list all plugins
        let { data } = await this.request.json( this.api_url + '/app/get_plugins/v1', {} );
        assert.ok( data.code === 0, 'successful api response' );
        assert.ok( Array.isArray(data.rows), 'expected rows array' );
        assert.ok( data.list && (data.list.length >= 0), 'expected list metadata' );
        // sanity check that built-in shell plugin exists
        assert.ok( Tools.findObject(data.rows, { id: 'shellplug' } ), 'expected shellplug to exist' );
    },

    async function test_api_get_plugin_missing_param(test) {
        // missing id param
        let { data } = await this.request.json( this.api_url + '/app/get_plugin/v1', {} );
        assert.ok( !!data.code, 'expected error for missing id' );
    },

    async function test_api_get_plugin_missing(test) {
        // non-existent plugin
        let { data } = await this.request.json( this.api_url + '/app/get_plugin/v1', { id: 'nope' } );
        assert.ok( !!data.code, 'expected error for missing plugin' );
    },

    async function test_api_create_plugin_missing_title(test) {
        // missing required title
        let { data } = await this.request.json( this.api_url + '/app/create_plugin/v1', {
            type: 'event'
        });
        assert.ok( !!data.code, 'expected error for missing title' );
    },

    async function test_api_create_plugin_missing_type(test) {
        // missing required type
        let { data } = await this.request.json( this.api_url + '/app/create_plugin/v1', {
            title: 'Bad Plugin'
        });
        assert.ok( !!data.code, 'expected error for missing type' );
    },

    async function test_api_create_plugin_invalid_type(test) {
        // invalid type value
        let { data } = await this.request.json( this.api_url + '/app/create_plugin/v1', {
            title: 'Bad Plugin 2',
            type: 'nope'
        });
        assert.ok( !!data.code, 'expected error for invalid type' );
    },

    async function test_api_create_plugin(test) {
        // create new plugin
        let { data } = await this.request.json( this.api_url + '/app/create_plugin/v1', {
            title: 'Unit Test Plugin',
            enabled: true,
            type: 'event',
            command: '[shell-plugin]',
            params: [ { id: 'foo', type: 'text', title: 'Foo', value: '' } ],
            notes: 'Created by unit tests'
        });
        assert.ok( data.code === 0, 'successful api response' );
        assert.ok( data.plugin && data.plugin.id, 'expected plugin in response' );
        this.plugin_id = data.plugin.id;
    },

    async function test_api_get_new_plugin(test) {
        // fetch our new plugin
        let { data } = await this.request.json( this.api_url + '/app/get_plugin/v1', { id: this.plugin_id } );
        assert.ok( data.code === 0, 'successful api response' );
        assert.ok( data.plugin && data.plugin.id === this.plugin_id, 'plugin id unexpected' );
        assert.ok( data.plugin.title === 'Unit Test Plugin', 'unexpected plugin title' );
        assert.ok( Array.isArray(data.plugin.params), 'expected params array' );
        assert.ok( !!Tools.findObject(data.plugin.params, { id: 'foo' }), 'expected foo param present' );
    },

    async function test_api_update_plugin_missing_id(test) {
        // update without id should error
        let { data } = await this.request.json( this.api_url + '/app/update_plugin/v1', { title: 'oops' } );
        assert.ok( !!data.code, 'expected error for missing id' );
    },

    async function test_api_update_plugin_invalid_params(test) {
        // invalid param id should error
        let { data } = await this.request.json( this.api_url + '/app/update_plugin/v1', {
            id: this.plugin_id,
            params: [ { id: '123bad', type: 'text', title: 'Bad', value: '' } ]
        });
        assert.ok( !!data.code, 'expected error for invalid param id' );
    },

    async function test_api_update_plugin(test) {
        // update our plugin
        let { data } = await this.request.json( this.api_url + '/app/update_plugin/v1', {
            id: this.plugin_id,
            title: 'UTP v2',
            enabled: false
        });
        assert.ok( data.code === 0, 'successful api response' );
    },

    async function test_api_get_updated_plugin(test) {
        // verify updates
        let { data } = await this.request.json( this.api_url + '/app/get_plugin/v1', { id: this.plugin_id } );
        assert.ok( data.code === 0, 'successful api response' );
        assert.ok( data.plugin && data.plugin.title === 'UTP v2', 'unexpected plugin title' );
        assert.ok( data.plugin.enabled === false, 'unexpected plugin enabled flag' );
    },

    async function test_api_delete_plugin_missing_id(test) {
        // delete without id should error
        let { data } = await this.request.json( this.api_url + '/app/delete_plugin/v1', {} );
        assert.ok( !!data.code, 'expected error for missing id' );
    },

    async function test_api_delete_plugin_nonexistent(test) {
        // delete non-existent plugin should error
        let { data } = await this.request.json( this.api_url + '/app/delete_plugin/v1', { id: 'nope' } );
        assert.ok( !!data.code, 'expected error for missing plugin' );
    },

    async function test_api_delete_plugin(test) {
        // delete our plugin
        let { data } = await this.request.json( this.api_url + '/app/delete_plugin/v1', { id: this.plugin_id } );
        assert.ok( data.code === 0, 'successful api response' );
    },

    async function test_api_get_plugin_deleted(test) {
        // ensure deleted
        let { data } = await this.request.json( this.api_url + '/app/get_plugin/v1', { id: this.plugin_id } );
        assert.ok( !!data.code, 'expected error for missing plugin' );
        delete this.plugin_id;
    }

];

