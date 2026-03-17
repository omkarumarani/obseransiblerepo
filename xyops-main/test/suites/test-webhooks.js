const assert = require('node:assert/strict');
const Tools = require('pixl-tools');

exports.tests = [

    async function test_api_get_web_hooks(test) {
        // list all web hooks
        let { data } = await this.request.json( this.api_url + '/app/get_web_hooks/v1', {} );
        assert.ok( data.code === 0, 'successful api response' );
        assert.ok( Array.isArray(data.rows), 'expected rows array' );
        assert.ok( data.list && (data.list.length >= 0), 'expected list metadata' );
    },

    async function test_api_get_web_hook_missing_param(test) {
        // missing id param
        let { data } = await this.request.json( this.api_url + '/app/get_web_hook/v1', {} );
        assert.ok( !!data.code, 'expected error for missing id' );
    },

    async function test_api_get_web_hook_missing(test) {
        // non-existent hook
        let { data } = await this.request.json( this.api_url + '/app/get_web_hook/v1', { id: 'nope' } );
        assert.ok( !!data.code, 'expected error for missing web hook' );
    },

    async function test_api_create_web_hook_missing_title(test) {
        // missing required title
        let { data } = await this.request.json( this.api_url + '/app/create_web_hook/v1', {
            method: 'POST',
            url: this.api_url + '/app/echo'
        });
        assert.ok( !!data.code, 'expected error for missing title' );
    },

    async function test_api_create_web_hook_invalid_url(test) {
        // invalid url (must be http/https)
        let { data } = await this.request.json( this.api_url + '/app/create_web_hook/v1', {
            title: 'Bad Hook',
            method: 'POST',
            url: 'ftp://example.com/hook'
        });
        assert.ok( !!data.code, 'expected error for invalid url' );
    },

    async function test_api_create_web_hook_invalid_macro(test) {
        // invalid JEXL macro in body should error
        let { data } = await this.request.json( this.api_url + '/app/create_web_hook/v1', {
            title: 'Bad Macro Hook',
            method: 'POST',
            url: this.api_url + '/app/echo',
            body: '{\n  "text": "{{ ### }}"\n}',
            timeout: 5
        });
        assert.ok( !!data.code, 'expected error for invalid body macro' );
    },

    async function test_api_create_web_hook(test) {
        // create new web hook that points back to our echo API
        let { data } = await this.request.json( this.api_url + '/app/create_web_hook/v1', {
            title: 'Unit Test Web Hook',
            enabled: true,
            url: this.api_url + '/app/echo',
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

    async function test_api_get_new_web_hook(test) {
        // fetch our new web hook
        let { data } = await this.request.json( this.api_url + '/app/get_web_hook/v1', { id: this.web_hook_id } );
        assert.ok( data.code === 0, 'successful api response' );
        assert.ok( data.web_hook && data.web_hook.id === this.web_hook_id, 'unexpected web_hook id' );
        assert.ok( data.web_hook.title === 'Unit Test Web Hook', 'unexpected web_hook title' );
        assert.ok( Array.isArray(data.web_hook.headers), 'expected headers array' );
    },

    async function test_api_update_web_hook_missing_id(test) {
        // update without id should error
        let { data } = await this.request.json( this.api_url + '/app/update_web_hook/v1', { title: 'oops' } );
        assert.ok( !!data.code, 'expected error for missing id' );
    },

    async function test_api_update_web_hook(test) {
        // update our web hook
        let { data } = await this.request.json( this.api_url + '/app/update_web_hook/v1', {
            id: this.web_hook_id,
            title: 'UTWH v2',
            enabled: false,
            timeout: 20,
            follow: true
        });
        assert.ok( data.code === 0, 'successful api response' );
    },

    async function test_api_get_updated_web_hook(test) {
        // verify updates
        let { data } = await this.request.json( this.api_url + '/app/get_web_hook/v1', { id: this.web_hook_id } );
        assert.ok( data.code === 0, 'successful api response' );
        assert.ok( data.web_hook && data.web_hook.title === 'UTWH v2', 'unexpected web_hook title' );
        assert.ok( data.web_hook.enabled === false, 'unexpected enabled flag' );
        assert.ok( data.web_hook.follow === true, 'unexpected follow flag' );
        assert.ok( data.web_hook.timeout === 20, 'unexpected timeout value' );
    },

    async function test_api_test_web_hook_missing_title(test) {
        // test requires title/method/url/id
        let { data } = await this.request.json( this.api_url + '/app/test_web_hook/v1', {
            id: this.web_hook_id,
            method: 'POST',
            url: this.api_url + '/app/echo'
        });
        assert.ok( !!data.code, 'expected error for missing title' );
    },

    async function test_api_test_web_hook(test) {
        // run live test against our echo API
        const msg = 'Hello from web hook test';
        let { data } = await this.request.json( this.api_url + '/app/test_web_hook/v1', {
            id: this.web_hook_id,
            title: 'Test Hook',
            method: 'POST',
            url: this.api_url + '/app/echo',
            headers: [ { name: 'Content-Type', value: 'application/json' } ],
            body: `{
  "text": "${msg}"
}`,
            timeout: 10
        });
        assert.ok( data.code === 0, 'successful api response' );
        assert.ok( data.result && data.result.code === 0, 'expected test success' );
        assert.ok( typeof data.result.details === 'string', 'expected markdown details' );
        assert.ok( data.result.details.includes('**Method:** POST'), 'expected method in details' );
        assert.ok( data.result.details.includes('**Response Body:**'), 'expected response body in details' );
        assert.ok( data.result.details.includes(msg), 'expected our message echoed back' );
    },

    async function test_api_delete_web_hook_missing_id(test) {
        // delete without id should error
        let { data } = await this.request.json( this.api_url + '/app/delete_web_hook/v1', {} );
        assert.ok( !!data.code, 'expected error for missing id' );
    },

    async function test_api_delete_web_hook_nonexistent(test) {
        // delete non-existent hook should error
        let { data } = await this.request.json( this.api_url + '/app/delete_web_hook/v1', { id: 'nope' } );
        assert.ok( !!data.code, 'expected error for missing web hook' );
    },

    async function test_api_delete_web_hook(test) {
        // delete our web hook
        let { data } = await this.request.json( this.api_url + '/app/delete_web_hook/v1', { id: this.web_hook_id } );
        assert.ok( data.code === 0, 'successful api response' );
    },

    async function test_api_get_web_hook_deleted(test) {
        // ensure deleted
        let { data } = await this.request.json( this.api_url + '/app/get_web_hook/v1', { id: this.web_hook_id } );
        assert.ok( !!data.code, 'expected error for missing web hook' );
        delete this.web_hook_id;
    }

];
