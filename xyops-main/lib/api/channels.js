// xyOps API Layer - Notification Channels
// Copyright (c) 2019 - 2026 PixlCore LLC
// Released under the BSD 3-Clause License.
// See the LICENSE.md file in this repository.

const fs = require('fs');
const assert = require("assert");
const async = require('async');
const Tools = require("pixl-tools");

class Channels {
	
	api_get_channels(args, callback) {
		// get list of all channels
		var self = this;
		var params = args.params;
		if (!this.requireMaster(args, callback)) return;
		
		this.loadSession(args, function(err, session, user) {
			if (err) return self.doError('session', err.message, callback);
			if (!self.requireValidUser(session, user, callback)) return;
			
			// return items and list header
			callback({
				code: 0,
				rows: self.channels,
				list: { length: self.channels.length }
			});
			
		} ); // loaded session
	}
	
	api_get_channel(args, callback) {
		// get single channel for editing
		var self = this;
		var params = Tools.mergeHashes( args.params, args.query );
		if (!this.requireMaster(args, callback)) return;
		
		if (!this.requireParams(params, {
			id: /^[a-z0-9_]+$/
		}, callback)) return;
		
		this.loadSession(args, function(err, session, user) {
			if (err) return self.doError('session', err.message, callback);
			if (!self.requireValidUser(session, user, callback)) return;
			
			var channel = Tools.findObject( self.channels, { id: params.id } );
			if (!channel) return self.doError('channel', "Failed to locate channel: " + params.id, callback);
			
			// success, return item
			callback({ code: 0, channel: channel });
			
		} ); // loaded session
	}
	
	api_create_channel(args, callback) {
		// add new channel
		var self = this;
		var params = args.params;
		if (!this.requireMaster(args, callback)) return;
		
		// auto-generate unique ID if not specified
		if (!params.id) params.id = Tools.generateShortID('ch');
		
		if (!this.requireParams(params, {
			id: /^[a-z0-9_]+$/,
			title: /\S/
		}, callback)) return;
		
		this.loadSession(args, function(err, session, user) {
			if (err) return self.doError('session', err.message, callback);
			if (!self.requireValidUser(session, user, callback)) return;
			if (!self.requirePrivilege(user, 'create_channels', callback)) return;
			
			args.user = user;
			args.session = session;
			
			params.username = user.username || user.id;
			params.created = params.modified = Tools.timeNow(true);
			params.revision = 1;
			
			// channel id must be unique
			if (Tools.findObject(self.channels, { id: params.id })) {
				return self.doError('channel', "That Channel ID already exists: " + params.id, callback);
			}
			
			self.logDebug(6, "Creating new channel: " + params.title, params);
			
			self.storage.listPush( 'global/channels', params, function(err) {
				if (err) {
					return self.doError('channel', "Failed to create channel: " + err, callback);
				}
				
				self.logDebug(6, "Successfully created channel: " + params.title, params);
				self.logTransaction('channel_create', params.title, self.getClientInfo(args, { channel: params, keywords: [ params.id ] }));
				
				// add to in-memory cache
				self.channels.push( Tools.copyHash(params, true) );
				
				// send api response
				callback({ code: 0, channel: params });
				
				// notify all users
				self.doUserBroadcastAll('update', { channels: self.channels });
				
			} ); // listPush
		} ); // loadSession
	}
	
	api_update_channel(args, callback) {
		// update existing channel
		var self = this;
		var params = args.params;
		if (!this.requireMaster(args, callback)) return;
		
		if (!this.requireParams(params, {
			id: /^[a-z0-9_]+$/
		}, callback)) return;
		
		this.loadSession(args, function(err, session, user) {
			if (err) return self.doError('session', err.message, callback);
			if (!self.requireValidUser(session, user, callback)) return;
			if (!self.requirePrivilege(user, 'edit_channels', callback)) return;
			
			args.user = user;
			args.session = session;
			
			params.modified = Tools.timeNow(true);
			params.revision = "+1";
			
			self.logDebug(6, "Updating channel: " + params.id, params);
			
			self.storage.listFindUpdate( 'global/channels', { id: params.id }, params, function(err, channel) {
				if (err) {
					return self.doError('channel', "Failed to update channel: " + err, callback);
				}
				
				self.logDebug(6, "Successfully updated channel: " + channel.title, params);
				self.logTransaction('channel_update', channel.title, self.getClientInfo(args, { channel: channel, keywords: [ params.id ] }));
				
				// update in-memory cache
				Tools.mergeHashInto( Tools.findObject( self.channels, { id: params.id } ) || {}, channel );
				
				// send api response
				callback({ code: 0 });
				
				// notify all users
				self.doUserBroadcastAll('update', { channels: self.channels });
				
			} ); // listFindUpdate
		} ); // loadSession
	}
	
	api_delete_channel(args, callback) {
		// delete existing channel
		var self = this;
		var params = args.params;
		if (!this.requireMaster(args, callback)) return;
		
		if (!this.requireParams(params, {
			id: /^[a-z0-9_]+$/
		}, callback)) return;
		
		this.loadSession(args, function(err, session, user) {
			if (err) return self.doError('session', err.message, callback);
			if (!self.requireValidUser(session, user, callback)) return;
			if (!self.requirePrivilege(user, 'delete_channels', callback)) return;
			
			args.user = user;
			args.session = session;
			
			self.logDebug(6, "Deleting channel: " + params.id, params);
			
			self.storage.listFindDelete( 'global/channels', { id: params.id }, function(err, channel) {
				if (err) {
					return self.doError('channel', "Failed to delete channel: " + err, callback);
				}
				
				self.logDebug(6, "Successfully deleted channel: " + channel.title, channel);
				self.logTransaction('channel_delete', channel.title, self.getClientInfo(args, { channel: channel, keywords: [ params.id ] }));
				
				// remove from in-memory cache
				Tools.deleteObject( self.channels, { id: params.id } );
				
				// send api response
				callback({ code: 0 });
				
				// notify all users
				self.doUserBroadcastAll('update', { channels: self.channels });
				
			} ); // listFindDelete
		} ); // loadSession
	}
	
}; // class Channels

module.exports = Channels;
