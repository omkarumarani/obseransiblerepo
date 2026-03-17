const assert = require('node:assert/strict');
const Tools = require('pixl-tools');

exports.tests = [

    async function test_api_get_secrets(test) {
        // list all secrets (metadata only)
        let { data } = await this.request.json( this.api_url + '/app/get_secrets/v1', {} );
        assert.ok( data.code === 0, 'successful api response' );
        assert.ok( Array.isArray(data.rows), 'expected rows array' );
        assert.ok( data.list && (data.list.length >= 0), 'expected list metadata' );
    },

    async function test_api_get_secret_missing_param(test) {
        // missing id param should error
        let { data } = await this.request.json( this.api_url + '/app/get_secret/v1', {} );
        assert.ok( !!data.code, 'expected error for missing id' );
    },

    async function test_api_create_secret_missing_title(test) {
        // missing required title
        let { data } = await this.request.json( this.api_url + '/app/create_secret/v1', { enabled: true } );
        assert.ok( !!data.code, 'expected error for missing title' );
    },

    async function test_api_create_secret(test) {
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
            // use built-in plugin and the final category created by category suite (if present)
            plugins: ['shellplug'],
            categories: this.category_final_id ? [ this.category_final_id ] : [],
            events: [],
            web_hooks: ['example_hook'],
            fields
        });
        assert.ok( data.code === 0, 'successful api response' );
        assert.ok( data.secret && data.secret.id, 'expected secret in response' );
        assert.ok( Array.isArray(data.secret.names) && data.secret.names.length === 3, 'expected names array derived from fields' );
        assert.ok( data.secret.names.includes('DB_PASS'), 'expected DB_PASS in names' );
        this.secret_id = data.secret.id;
    },

    async function test_api_get_secret(test) {
        // fetch our new secret (metadata only)
        let { data } = await this.request.json( this.api_url + '/app/get_secret/v1', { id: this.secret_id } );
        assert.ok( data.code === 0, 'successful api response' );
        assert.ok( data.secret && data.secret.id === this.secret_id, 'secret id unexpected' );
        assert.ok( Array.isArray(data.secret.names) && data.secret.names[0] === 'DB_HOST', 'unexpected names content' );
        assert.ok( Array.isArray(data.secret.plugins) && data.secret.plugins.includes('shellplug'), 'expected shellplug assignment' );
    },

    async function test_api_decrypt_secret(test) {
        // decrypt our secret (admin only)
        let { data } = await this.request.json( this.api_url + '/app/decrypt_secret/v1', { id: this.secret_id } );
        assert.ok( data.code === 0, 'successful api response' );
        assert.ok( Array.isArray(data.fields) && data.fields.length === 3, 'expected fields array' );
        assert.ok( !!Tools.findObject(data.fields, { name: 'DB_PASS', value: 'CorrectHorseBatteryStaple' }), 'expected DB_PASS value' );
    },

    async function test_api_update_secret_missing_id(test) {
        // update without id should error
        let { data } = await this.request.json( this.api_url + '/app/update_secret/v1', { title: 'oops' } );
        assert.ok( !!data.code, 'expected error for missing id' );
    },

    async function test_api_update_secret_metadata(test) {
        // update metadata only
        let { data } = await this.request.json( this.api_url + '/app/update_secret/v1', {
            id: this.secret_id,
            title: 'UTS v2',
            enabled: false,
            notes: 'updated by tests'
        });
        assert.ok( data.code === 0, 'successful api response' );
    },

    async function test_api_get_updated_secret(test) {
        // verify metadata changes
        let { data } = await this.request.json( this.api_url + '/app/get_secret/v1', { id: this.secret_id } );
        assert.ok( data.code === 0, 'successful api response' );
        assert.ok( data.secret && data.secret.title === 'UTS v2', 'unexpected secret title' );
        assert.ok( data.secret.enabled === false, 'unexpected secret enabled flag' );
    },

    async function test_api_update_secret_fields(test) {
        // replace variables (fields) and ensure names regenerate
        const newFields = [
            { name: 'API_KEY', value: 'abc123' },
            { name: 'API_URL', value: 'https://api.local' }
        ];
        let { data } = await this.request.json( this.api_url + '/app/update_secret/v1', {
            id: this.secret_id,
            fields: newFields
        });
        assert.ok( data.code === 0, 'successful api response' );
    },

    async function test_api_decrypt_secret_after_update(test) {
        // decrypt again and verify updated values
        let { data } = await this.request.json( this.api_url + '/app/decrypt_secret/v1', { id: this.secret_id } );
        assert.ok( data.code === 0, 'successful api response' );
        assert.ok( Array.isArray(data.fields) && data.fields.length === 2, 'expected 2 fields after update' );
        assert.ok( !!Tools.findObject(data.fields, { name: 'API_KEY', value: 'abc123' }), 'expected API_KEY value' );
        assert.ok( !!Tools.findObject(data.fields, { name: 'API_URL', value: 'https://api.local' }), 'expected API_URL value' );
    },

    async function test_api_delete_secret_missing_id(test) {
        // delete without id should error
        let { data } = await this.request.json( this.api_url + '/app/delete_secret/v1', {} );
        assert.ok( !!data.code, 'expected error for missing id' );
    },

    async function test_api_delete_secret_nonexistent(test) {
        // delete non-existent secret should error
        let { data } = await this.request.json( this.api_url + '/app/delete_secret/v1', { id: 'nope' } );
        assert.ok( !!data.code, 'expected error for missing secret' );
    },

    async function test_api_delete_secret(test) {
        // delete our secret
        let { data } = await this.request.json( this.api_url + '/app/delete_secret/v1', { id: this.secret_id } );
        assert.ok( data.code === 0, 'successful api response' );
    },

    async function test_api_get_secret_deleted(test) {
        // ensure deleted
        let { data } = await this.request.json( this.api_url + '/app/get_secret/v1', { id: this.secret_id } );
        assert.ok( !!data.code, 'expected error for missing secret' );
        delete this.secret_id;
    }

];

