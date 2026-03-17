// xyOps API Layer - Documentation view and search
// Copyright (c) 2019 - 2026 PixlCore LLC
// Released under the BSD 3-Clause License.
// See the LICENSE.md file in this repository.

const fs = require('fs');
const Path = require('path');
const async = require('async');
const Tools = require("pixl-tools");

class Docs {
	
	api_get_doc(args, callback) {
		// fetch raw markdown for doc
		var self = this;
		var params = args.query;
		
		if (!this.requireParams(params, {
			doc: /^\w+$/
		}, callback)) return;
		
		if (params.doc == 'search') {
			return this.api_search_docs(args, callback);
		}
		
		var file = `docs/${params.doc}.md`;
		fs.readFile( file, 'utf8', function(err, text) {
			if (err) return callback({ code: 1, description: '' + err });
			
			self.setCacheResponse(args, self.config.get('ttl'));
			callback({ code: 0, text });
		} );
	}
	
	api_search_docs(args, callback) {
		// perform brute-force substring search across docs
		var self = this;
		var params = args.query;
		
		if (!this.requireParams(params, {
			anchor: /.+/
		}, callback)) return;
		
		if (!params.limit) params.limit = 100;
		
		try { params.anchor = decodeURIComponent(params.anchor); }
		catch (err) { return callback({ code: 1, description: "Invalid search query." }); }
		
		params.anchor = params.anchor.toString().replace(/<[^>]*>/g, '');
		if (!params.anchor.match(/\S/)) {
			return callback({ code: 1, description: "Invalid search query." });
		}
		
		var lower_query = params.anchor.toLowerCase();
		var matches = [];
		
		Tools.glob( 'docs/*.md', function(err, files) {
			if (err) return callback({ code: 1, description: '' + err });
			
			async.eachSeries( files,
				function(file, callback) {
					// process file
					var doc_id = Path.basename(file).replace(/\.\w+$/, '');
					
					fs.readFile( file, 'utf8', function(err, text) {
						if (err) return callback(err);
						
						// grab title from first level-1 header
						if (!text.match(/^\#\s+([^\n]+)/)) return callback();
						var title = RegExp.$1;
						var in_code_block = false;
						var last_title = '';
						var last_anchor = '';
						var lines = text.trim().split(/\n/);
						lines.shift(); // exclude title from search results
						
						lines.forEach( function(line) { 
							if (line.match(/^\`\`\`/)) in_code_block = !in_code_block;
							
							var in_heading = false;
							if (!in_code_block && line.match(/^(\#+)\s+(.+)$/)) {
								last_title = RegExp.$2;
								last_anchor = last_title.trim().replace(/\W+/g, '-').toLowerCase();
								in_heading = true;
							}
							
							var idx = line.toLowerCase().indexOf(lower_query);
							if (idx > -1) {
								var href = doc_id + '.md';
								if (last_anchor) href += '#' + last_anchor;
								matches.push({ 
									doc: doc_id, 
									title, line, idx, 
									code: in_code_block, 
									section: last_title, 
									anchor: last_anchor,
									heading: in_heading,
									href
								});
							}
						} );
						
						if (matches.length >= params.limit) callback("STOP");
						else callback();
					} ); // fs.readFile
				},
				function(err) {
					// format results as markdown
					self.setCacheResponse(args, self.config.get('ttl'));
					
					var more = (err === 'STOP') ? '+' : '';
					var text = '';
					text += `# Search Results\n\n`;
					
					if (!matches.length) {
						text += `No results found for &ldquo;${params.anchor}&rdquo;.  Please try a different search query.`;
						return callback({ code: 0, text });
					}
					
					text += `## ${Tools.commify(matches.length)}${more} ${Tools.pluralize('result', matches.length)} for &ldquo;${params.anchor}&rdquo;:\n`;
					
					var last_href = '';
					
					matches.forEach( function(match) {
						if (match.href != last_href) {
							last_href = match.href;
							text += `\n- **[${match.title}](${match.doc}.md)**`;
							if (match.anchor) text += ` → **[${match.section}](${match.href})**`;
							text += "\n";
						}
						
						// if search matched a heading, do not include a preview
						if (match.heading) return;
						
						// cleanup and sanitization
						match.line = match.line.replace(/</g, '&lt;').replace(/>/g, '&gt;').trim();
						if (match.code) match.line = '<code>' + match.line.replace(/\`/g, '') + '</code>';
						
						var preview = match.line.replace(/^\s*(\-|\d+\.|\#+)\s+/, '').trim();
						text += `\t- ${preview}\n`;
					} );
					
					if (more) {
						text += `\n*(Additional matches were chopped.)*`;
					}
					
					callback({ code: 0, text });
				}
			); // eachSeries
		} ); // glob
	}
	
}; // class Docs

module.exports = Docs;
