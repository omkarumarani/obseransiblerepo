# xyOps Backup Format

## Overview

This document describes the xyOps Backup Format (XYBK) v1.0, used to bulk export and import data from a xyOps system. The format supports selecting categories of data (lists, database indexes, and extras), or including everything. Files are [NDJSON](https://github.com/ndjson/ndjson-spec) with support for comment lines and blank lines, and are typically wrapped in [Gzip](https://en.wikipedia.org/wiki/Gzip) for transport.

- **Title**: xyOps Backup Format
- **ID**: XYBK
- **Version**: 1.0
- **Date**: December 12, 2025
- **Authors**: Joseph Huckaby (PixlCore)

XYBK is primarily consumed by the Admin "Export Data" and "Import Data" features. The exporter streams a Gzip-compressed NDJSON file to the client, and the importer accepts either plain NDJSON or Gzip-wrapped NDJSON.

## File Structure

An XYBK file is a sequence of UTF-8 text lines. Three types of lines are allowed:

- **Comment**: Any line beginning with `#` is a comment and ignored by the importer.
- **Blank**: Empty or whitespace-only lines are allowed and ignored.
- **Record**: A single JSON object on one line (NDJSON). These are processed in order.

The file begins with a short header emitted as a comment block (for human readability only):

```
# xyOps Data Export v1.0
# Hostname: [host]
# Date/Time: [string]
# Format: NDJSON
```

Following the header, the file contains one or more labeled sections (comment lines) and NDJSON records. Section headers are advisory and ignored by the importer. The importer only processes JSON lines that start with `{`.

## Record Types

Each NDJSON record must be exactly one of the following forms:

### Storage Put

```json
{ "key": "<storage_key>", "value": /* json_or_base64 */ }
```

- Writes directly to [pixl-server-storage](https://github.com/jhuckaby/pixl-server-storage) as a key/value "put".
- For binary keys, `value` contains a Base64 string. On import, binary detection is automatic via key pattern, and the value is decoded back to raw bytes.
- For JSON keys, `value` is a JSON object which is stored as-is.

### Storage Command

```json
{ "cmd": "<method>", "args": [ /* arg1, arg2, ... */ ] }
```

- Invokes a storage API on [pixl-server-storage](https://github.com/jhuckaby/pixl-server-storage), e.g. `listDelete`.
- Arguments are passed as-is. The importer appends its own callback internally.
- Used by exports to prepare state for re-population (e.g. delete list pages before re-creating them).

### Database Record

```json
{ "index": "<index_id>", "id": "<record_id>", "record": { /* ... */ } }
```

- Inserts a database record into [Unbase](https://github.com/jhuckaby/pixl-server-unbase) via `unbase.insert(index, id, record)`.
- Semantics are "create or replace" by ID.

## Sections

The exporter adds comment section headers to group related lines. These are informational only and ignored during import. You may encounter the following section headers:

- `# List: <key>`
- `# Database Index: <index> (<query>)`
- `# User Data:`
- `# Bucket Data`
- `# Bucket Files`
- `# Encrypted Secret Data`
- `# Job Files (<query>)`
- `# Ticket Files (<query>)`
- `# Monitor Timeline Data (<query>)`

### Lists

Many xyOps subsystems are modeled as storage "Lists" (paged arrays). See [Lists](https://github.com/jhuckaby/pixl-server-storage/blob/master/docs/Lists.md) for list internals. Exports include list metadata and all pages. The format is:

Pre-delete list to make way for the incoming one:

```json
{ "cmd": "listDelete", "args": [ "<key>", false ] }
```

Add the list header (key/value):

```json
{ "key": "<key>", "value": { "page_size": 100, "first_page": 1, "last_page": 5, "length": 500, "type": "list" } }
```

Add the list pages (key/value):

```json
{ "key": "<key>/<page>", "value": { "type": "list_page", "items": [ /* ... */ ] } }
```

The `items` array contains the actual list items. Pages are emitted from `first_page` to `last_page` inclusive.

Typical list keys exported under `global/` include:

- `alerts`, `api_keys`, `buckets`, `categories`, `channels`, `events`, `groups`, `monitors`, `plugins`, `secrets`, `tags`, `users`, `roles`, `web_hooks`

Note: User account records themselves are not stored in `global/users` (that list holds the roster). Actual user records are exported under `users/<username>` (see "User Data" below).

### Database Indexes

Exports can include full Unbase indexes, optionally filtered by a query. Each record is emitted as:

```json
{ "index": "<index_id>", "id": "<record_id>", "record": { /* ... */ } }
```

Common index IDs include: `alerts`, `jobs`, `servers`, `snapshots`, `activity`, `tickets`.

See [Unbase](https://github.com/jhuckaby/pixl-server-unbase) for more details.

### User Data

User account records are exported as storage keys:

- `users/<normalized_username>` → `{ ...user record... }`

If the "User Avatars" extra is selected, the following binary keys may also be included (Base64 values):

- `users/<normalized_username>/avatar/64.png`
- `users/<normalized_username>/avatar/256.png`

Passwords in user records are stored as salted [bcrypt](https://en.wikipedia.org/wiki/Bcrypt) hashes and are exported as stored.

### Buckets

If the Buckets list is selected, the exporter also includes bucket data and may include file payloads, depending on extras:

- `key: buckets/<bucket_id>/data` (JSON) containing the per-bucket metadata/data object.
- `key: buckets/<bucket_id>/files` (JSON) containing each file payload keyed by its storage path. File payloads are Base64.

### Secrets

Secret vault metadata lives in the `global/secrets` list (exported like any list). The secret payloads themselves are exported under:

- `key: secrets/<secret_id>` → Value is the encrypted blob (as stored). Contents are Base64-encoded encrypted data; secrets are not exported in plaintext.

### Job and Ticket Files/Logs

If selected via extras, job and ticket attachments are exported by key with Base64 payloads. For jobs, the compressed log may also be exported:

- Job files: `key: <file_path>` for each file in a job’s `files[]` list (subject to max size).
- Job log: `key: logs/jobs/<job_id>/log.txt.gz` if present and under size limit.
- Ticket files: `key: <file_path>` for each file in a ticket’s `files[]` list (subject to max size).

### Monitor Timeline Data

Server monitor time-series are stored as lists under `timeline/<server_id>/<system_id>`. When included, the exporter emits each timeline as a normal List (see [Lists](#lists)).

## Data Selection

The UI exposes three selection groups which map to exported item types:

- **Lists**: One or more of the standard lists under `global/` (see above). Choosing `users` also triggers "User Data" export for `users/<username>` records. Choosing `buckets` triggers "Bucket Data". Choosing `secrets` triggers "Encrypted Secret Data".
- **Indexes**: One or more Unbase indexes by ID (optionally filtered by query).
- **Extras**: Optional payloads and time-series:
  - `job_files`
  - `job_logs`
  - `bucket_files`
  - `ticket_files`
  - `monitor_data`
  - `stat_data` 
  - `user_avatars`

The exporter may be instructed to include "all" in any group. Internally, these selections are expanded into a stream of the record types described above.

## Compression

Exports are streamed as Gzip files with a filename like `xyops-data-export-YYYY-MM-DD-<id>.txt.gz`. The importer accepts either a plain `.txt` NDJSON file or a Gzip-compressed `.txt.gz` file.

## Security Characteristics

- **API Keys**: Only salted hashes are exported; plaintext API key material is never emitted. The `key` field is a salted SHA-256 digest stored at creation time.
- **Secrets**: Secret payloads are exported as encrypted blobs (Base64); plaintext is never exported.
- **Users**: Passwords are stored and exported as salted bcrypt hashes. No plaintext passwords are exported.

## Example

Snippet showing list export, API keys, and a database record:

```
# xyOps Data Export v1.0
# Hostname: joemax.xyops.io
# Date/Time: Tue Nov 18 2025 12:01:27 GMT-0800 (Pacific Standard Time)
# Format: NDJSON

# List: global/alerts
{"cmd":"listDelete","args":["global/alerts",false]}
{"key":"global/alerts","value":{"page_size":100,"first_page":0,"last_page":0,"length":5,"type":"list"}}
{"key":"global/alerts/0","value":{"type":"list_page","items":[{"id":"load_avg_high","title":"High CPU Load", ...}]}}

# List: global/api_keys
{"cmd":"listDelete","args":["global/api_keys",false]}
{"key":"global/api_keys","value":{"page_size":100,"first_page":0,"last_page":0,"length":2,"type":"list"}}
{"key":"global/api_keys/0","value":{"type":"list_page","items":[{"key":"<salted_sha256>","active":1, ...}]}}

# Database Index: tickets (*)
{"index":"tickets","id":"tmhzbmbagig","record":{"subject":"Alert: High Active Jobs on raspberrypi", ...}}
```

## Parsing Rules

- Comments and blank lines are ignored. Only lines that begin with `{` are parsed.
- Lines are processed in order. Commands may prepare state (e.g. list deletion) before subsequent puts.
- Key/value records are written via `storage.put(key, value)`. Binary keys are automatically Base64-decoded on import.
- Database records are inserted via `unbase.insert(index, id, record)`.
- Storage commands call the named method on the storage engine with provided args.
- The importer streams and validates line-by-line, collecting up to 100 errors for reporting, and continues past non-fatal errors.

## Lists and Storage Notes

xyOps stores most configuration objects as Lists. Useful references:

- [pixl-server-storage](https://github.com/jhuckaby/pixl-server-storage)
- [Lists](https://github.com/jhuckaby/pixl-server-storage/blob/master/docs/Lists.md)
- [Unbase](https://github.com/jhuckaby/pixl-server-unbase)

## Versioning

This document specifies XYBK v1.0. The exporter emits `# xyOps Data Export v1.0` in the header. Future versions may add new section headers and record shapes, but importers ignore comments and only rely on the three NDJSON record types defined here.
