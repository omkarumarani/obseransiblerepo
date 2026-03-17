# Database

This document describes the xyOps database schema. It lists every index (table), all indexed columns, and the dedicated sorters used for ordering results.

## Overview

xyOps uses [Unbase](https://github.com/jhuckaby/pixl-server-unbase) which sits on top of [pixl-server-storage](https://github.com/jhuckaby/pixl-server-storage) and its [Indexer](https://github.com/jhuckaby/pixl-server-storage/blob/master/docs/Indexer.md) subsystem. Records are stored as JSON in a key/value backend (SQLite by default), and Unbase builds searchable indexes and sorters from configured field definitions. Queries support both simple "field:words" and a structured [PxQL](https://github.com/jhuckaby/pixl-server-storage/blob/master/docs/Indexer.md#pxql-queries) syntax.

Notes:

- Type refers to the indexer type: word (default full-text/word), number, or date. Unless specified, a field is a word index.
- Date/number fields may be stored with reduced precision for performance (e.g., divided by 3600 to index hour-level time buckets).

## Jobs

Completed job records (see [Job](data.md#job)).

Indexed Columns:

| Column ID | Source | Type | Description |
|-----------|--------|------|-------------|
| `code` | [Job.code](data.md#job-code) | Word | Result code for the job (0 success, non-zero failure; special values like `warning`, `critical`, `abort`). |
| `date` | [Job.completed](data.md#job-completed) | Number | Completion timestamp indexed at hour precision. |
| `source` | [Job.source](data.md#job-source) | Word | Launch source (scheduler, plugin, key, user, action, alert, workflow). |
| `tags` | [Job.tags](data.md#event-tags) | Word | Tags assigned to the job. |
| `event` | [Job.event](data.md#job-event) | Word | Event ID that spawned the job. |
| `category` | [Job.category](data.md#event-category) | Word | Event category ID copied into the job. |
| `plugin` | [Event.plugin](data.md#event-plugin) | Word | Plugin ID that executed the job. |
| `server` | [Job.server](data.md#job-server) | Word | Server ID selected to run the job. |
| `groups` | [Job.groups](data.md#job-groups) | Word | Server group IDs copied into the job. |
| `workflow` | [Job.workflow](data.md#job-workflow) | Word | When part of a workflow, the workflow event ID. |
| `tickets` | [Job](data.md#job-tickets) | Word | Linked ticket IDs associated with the job. |

Sorters:

| Sorter ID | Source | Type | Description |
|-----------|--------|------|-------------|
| `completed` | [Job.completed](data.md#job-completed) | Number | Sort by job completion timestamp. |
| `elapsed` | [Job.elapsed](data.md#job-elapsed) | Number | Sort by job elapsed duration (seconds). |

## Alerts

Alert invocation records (see [AlertInvocation](data.md#alertinvocation)).

Indexed Columns:

| Column ID | Source | Type | Description |
|-----------|--------|------|-------------|
| `active` | [AlertInvocation.active](data.md#alertinvocation-active) | Word | Whether the alert is currently active (`true` or `false`). |
| `alert` | [AlertInvocation.alert](data.md#alertinvocation-alert) | Word | Alert definition ID. |
| `groups` | [AlertInvocation.groups](data.md#alertinvocation-groups) | Word | Groups the server belongs to. |
| `server` | [AlertInvocation.server](data.md#alertinvocation-server) | Word | Server ID associated with the invocation. |
| `start` | [AlertInvocation.date](data.md#alertinvocation-date) | Number | Start timestamp indexed at hour precision. |
| `end` | [AlertInvocation.modified](data.md#alertinvocation-modified) | Number | Last modified time indexed at hour precision. |
| `jobs` | [AlertInvocation.jobs](data.md#alertinvocation-jobs) | Word | Related job IDs. |
| `tickets` | [AlertInvocation.tickets](data.md#alertinvocation-tickets) | Word | Related ticket IDs. |

## Snapshots

Server and group snapshot records (see [Snapshot](data.md#snapshot)).

Indexed Columns:

| Column ID | Source | Type | Description |
|-----------|--------|------|-------------|
| `type` | [Snapshot.type](data.md#snapshot-type) | Word | Snapshot type: `server` or `group`. |
| `source` | [Snapshot.source](data.md#snapshot-source) | Word | Snapshot origin: `alert`, `watch`, `user`, or `job`. |
| `server` | [Snapshot.server](data.md#snapshot-server) | Word | Server ID for server snapshots. |
| `groups` | [Snapshot.groups](data.md#snapshot-groups) | Word | Groups associated at the time of the snapshot. |
| `date` | [Snapshot.date](data.md#snapshot-date) | Number | Snapshot timestamp indexed at hour precision. |
| `alerts` | [Snapshot.alerts](data.md#snapshot-alerts) | Word | Active alert invocation IDs at snapshot time. |
| `jobs` | [Snapshot.jobs](data.md#snapshot-jobs) | Word | Active job IDs at snapshot time. |

## Servers

Server records (see [Server](data.md#server)).

Indexed Columns:

| Column ID | Source | Type | Description |
|-----------|--------|------|-------------|
| `groups` | [Server.groups](data.md#server-groups) | Word | Group IDs (master list enabled). |
| `created` | [Server.created](data.md#server-created) | Number | Created timestamp indexed at hour precision. |
| `modified` | [Server.modified](data.md#server-modified) | Number | Last modified timestamp indexed at hour precision. |
| `keywords` | [Server.keywords](data.md#server-keywords) | Word | Search keywords (min 1, max 64 chars per word). |
| `os_platform` | [Server.info.os.platform](data.md#server-info) | Word | OS platform (filtered alphanumeric; master list/labels). |
| `os_distro` | [Server.info.os.distro](data.md#server-info) | Word | OS distribution (filtered alphanumeric; master list/labels). |
| `os_release` | [Server.info.os.release](data.md#server-info) | Word | OS release/version (filtered alphanumeric; master list/labels). |
| `os_arch` | [Server.info.os.arch](data.md#server-info) | Word | CPU architecture (filtered alphanumeric; master list/labels). |
| `cpu_virt` | [Server.info.virt.vendor](data.md#server-info) | Word | Virtualization vendor (filtered alphanumeric; master list/labels). |
| `cpu_brand` | [Server.info.cpu.combo](data.md#server-info) | Word | CPU vendor/brand (filtered alphanumeric; master list/labels). |
| `cpu_cores` | [Server.info.cpu.cores](data.md#server-info) | Word | CPU core count (filtered alphanumeric; master list/labels). |

## Activity

User/system activity log (see [Activity](data.md#activity)).

Indexed Columns:

| Column ID | Source | Type | Description |
|-----------|--------|------|-------------|
| `action` | [Activity.action](data.md#activity-action) | Word | Activity action identifier. |
| `keywords` | [Activity.keywords](data.md#activity-keywords) | Word | Keywords for search (IDs, usernames, IPs). |
| `date` | [Activity.epoch](data.md#activity-epoch) | Number | Activity timestamp indexed at hour precision. |

## Tickets

Ticket records (see [Ticket](data.md#ticket)).

Indexed Columns:

| Column ID | Source | Type | Description |
|-----------|--------|------|-------------|
| `type` | [Ticket.type](data.md#ticket-type) | Word | Ticket type (`issue`, `feature`, `change`, `maintenance`, `question`, `other`). |
| `num` | [Ticket.num](data.md#ticket-num) | Number | Auto-assigned ticket number. |
| `status` | [Ticket.status](data.md#ticket-status) | Word | Ticket status (`open`, `closed`, `draft`). |
| `category` | [Ticket.category](data.md#ticket-category) | Word | Category ID. |
| `username` | [Ticket.username](data.md#ticket-username) | Word | Creator username (filtered alphanumeric). |
| `assignees` | [Ticket.assignees](data.md#ticket-assignees) | Word | Assigned users (filtered alphanumeric array). |
| `cc` | [Ticket.cc](data.md#ticket-cc) | Word | Users CC’d (filtered alphanumeric array). |
| `jobs` | [Ticket](data.md#ticket) | Word | Related job IDs. |
| `tags` | [Ticket.tags](data.md#ticket-tags) | Word | Tag IDs (master list enabled). |
| `created` | [Ticket.created](data.md#ticket-created) | Number | Created timestamp indexed at hour precision. |
| `modified` | [Ticket.modified](data.md#ticket-modified) | Number | Last modified timestamp indexed at hour precision. |
| `due` | [Ticket.due](data.md#ticket-due) | Date | Due date. |
| `server` | [Ticket.server](data.md#ticket-server) | Word | Associated server ID. |
| `subject` | [Ticket.subject](data.md#ticket-subject) | Word | Short summary (FTS; stemming enabled). |
| `body` | [Ticket.body](data.md#ticket-body) | Word | Full-text search across username, subject and body (markdown filtered; stemming enabled). |
| `changes` | [Ticket.changes](data.md#ticket-changes) | Word | Full-text search across change log content (markdown filtered; stemming enabled). |

Sorters:

| Sorter ID | Source | Type | Description |
|-----------|--------|------|-------------|
| `num` | [Ticket.num](data.md#ticket-num) | Number | Sort by ticket number. |
| `modified` | [Ticket.modified](data.md#ticket-modified) | Number | Sort by last modified timestamp. |

## Column Properties

These are the common field properties supported by the indexer (see Indexer docs for full details):

- `id`: The column ID used in searches (e.g., `status:open`).
- `source`: Slash-delimited path to the source data field (can reference nested properties or multiple sources for FTS).
- `type`: Index type for the field or sorter. Omit for word indexes; may be `number` or `date` for fields, and `number` or `string` for sorters.
- `divide`: For numbers, divides the value before indexing. For dates, common value `3600` indexes at hour precision to improve performance.
- `min_word_length` / `max_word_length`: Bounds for token length in word indexes.
- `use_remove_words`: Toggle custom remove-word list.
- `use_stemmer`: Enable Porter stemming for word indexes.
- `filter`: Apply a filter prior to indexing (e.g., `alphanum`, `alphanum_array`, `markdown`).
- `master_list`: Maintain a master list of unique indexed values for quick summaries.
- `master_labels`: Maintain a master list of unique raw values (before filtering).
