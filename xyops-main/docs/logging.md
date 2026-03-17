# Logging

## Overview

This document explains how logging works in xyOps, lists every application log file, and shows an example line from each. xyOps uses the [pixl-server](https://github.com/jhuckaby/pixl-server) logging service (powered by [pixl-logger](https://github.com/jhuckaby/pixl-logger)) which writes plain text, `[bracket][delimited][columns]`.

## Log Format

Each log row is bracket-delimited with the following columns, in order:

| Log Column | Description |
|---|---|
| `hires_epoch` | High-resolution Unix time (floating point seconds). |
| `date` | Human-readable timestamp: `YYYY-MM-DD HH:MI:SS` (local server time). |
| `hostname` | Hostname of the server that wrote the log entry. |
| `pid` | Process ID (PID) of the writer process. |
| `component` | Component name that generated the entry. |
| `category` | One of `debug`, `transaction`, or `error`. |
| `code` | Debug level (1-10), transaction code, or error code. |
| `msg` | Message text. |
| `data` | Optional JSON data payload (may be empty). |

Example row:

```
[1763880095.551][2025-11-22 22:41:35][joemax.lan][10341][API][debug][9][Activating namespaced API handler: app/api_get_server for URI: /api/app/get_server][]
```

The column order is configurable via [log_columns](config.md#log_columns) (see config below), but the defaults above are recommended.

## Configuration Summary

The core logging settings live in `config.json` and defaults are:

```json
{
  "log_dir": "logs",
  "log_filename": "[component].log",
  "log_columns": ["hires_epoch", "date", "hostname", "pid", "component", "category", "code", "msg", "data"],
  "log_archive_path": "logs/archives/[yyyy]/[mm]/[dd]/[filename]-[yyyy]-[mm]-[dd].log.gz",
  "log_crashes": true,
  "debug_level": 9
}
```

If [log_dir](config.md#log_dir) is relative, it is computed based on the xyOps root directory (typically `/opt/xyops`).

The [log_filename](config.md#log_filename) property defaults to `[component].log` so each component writes to its own file. To use a single combined log, set a literal filename, for example:

```json
"log_filename": "xyops.log"
```

## Application Logs

Below are all current application log files, each with a short description and one real example row captured from this repository.

### Action.log
Actions fired by alerts, monitors, or job conditions (e.g., email, web hooks, plugins).

```
[1763841931.973][2025-11-22 12:05:31][joemax.lan][92938][Action][debug][8][Running job actions for condition: complete][{"job_id":"jmiapxwar79"}]
```

### API.log
Incoming HTTP API requests, routing, authorization, and handler activity.

```
[1763880095.55][2025-11-22 22:41:35][joemax.lan][10341][API][debug][6][Handling API request: GET /api/app/get_server?id=smf4j79snhe&cachebust=1763880046.493][]
```

### Comm.log
WebSocket client connections, disconnections, and page/activity updates.

```
[1763880034.062][2025-11-22 22:40:34][joemax.lan][10341][Comm][debug][6][User socket has authenticated successfully: wsmibcmostgu][{"username":"admin"}]
```

### Debug.log
Internal debug service messages and diagnostics.

```
[1763879905.013][2025-11-22 22:38:25][joemax.lan][10341][Debug][debug][3][Debug service listening for base URI: /internal/debug][]
```

### Error.log
All errors across the system (validation errors, storage issues, job failures, disconnects, etc.).

```
[1763879534.953][2025-11-22 22:32:14][joemax.lan][10176][Storage][error][rollback][Aborting transaction: 16][{"path":"timeline/smf4j79snhe/hourly","actions":0}]
```

### Filesystem.log
Events from the local filesystem storage engine (binary files, paths, open/close, etc.).

```
[1763879912.74][2025-11-22 22:38:32][joemax.lan][10341][Filesystem][debug][9][Fetching Binary Stream: users/admin/avatar/64.png][data/users/admin/avatar/64.png]
```

### Hybrid.log
Hybrid document/binary storage orchestration and lifecycle.

```
[1763879904.984][2025-11-22 22:38:24][joemax.lan][10341][Hybrid][debug][2][Setting up hybrid engine][{"docEngine":"SQLite","binaryEngine":"Filesystem"}]
```

### Job.log
Job creation, start/stop, and internal job lifecycle.

```
[1763879516.523][2025-11-22 22:31:56][joemax.lan][10176][Job][debug][5][Starting new internal job: imibcblgrjg][{"title":"Worker server upgrade","type":"maint","username":"admin","params":{"targets":["main"],"release":"latest","stagger":30},"stats":{"servers":0,"skipped":0},"details":"","id":"imibcblgrjg","started":1763879516.523,"progress":0}]
```

### Maint.log
Nightly maintenance and housekeeping tasks (daily stats reset, archival, cleanup).

```
[1763831742.113][2025-11-22 09:15:42][joemax.lan][92938][Maint][debug][6][A new day dawns, resetting daily stats.][]
```

### Monitor.log
Monitor evaluation, expressions, alert triggering/clearing, and data submission.

```
[1763880062.526][2025-11-22 22:41:02][joemax.lan][10341][Monitor][debug][9][Checking alert expression for raspberrypi/disk_usage_root_high: monitors.disk_usage_root >= 90][{"alert":{"id":"disk_usage_root_high","title":"Disk Full","expression":"monitors.disk_usage_root >= 90","message":"Root filesystem is {{pct(monitors.disk_usage_root)}} full.","groups":[],"actions":[],"monitor_id":"disk_usage_root","enabled":true,"samples":1,"notes":"","username":"admin","modified":1754365754,"created":1754365754,"revision":1},"server":"smf4j79snhe","hostname":"raspberrypi","expression":"monitors.disk_usage_root >= 90"}]
```

### Multi.log
Multi-server cluster status and conductor election.

```
[1763879905.018][2025-11-22 22:38:25][joemax.lan][10341][Multi][debug][1][We are becoming primary][{"id":"joemax.lan"}]
```

### Scheduler.log
Scheduler ticks, queue evaluation, and job due events.

```
[1763921040.048][2025-11-23 10:04:00][joemax.lan][14546][Scheduler][debug][5][Ticking scheduler for timestamp: Sun Nov 23 2025 10:04:00 GMT-0800 (Pacific Standard Time)][]
```

### Secret.log
Secret usage by plugins, hooks, and other components (no secret values are logged).

```
[1763880046.442][2025-11-22 22:40:46][joemax.lan][10341][Secret][debug][1][Using secret zmi94hfmspt (Dev Database) for plugins: pmibcla6mg8][{"secret":{"id":"zmi94hfmspt","title":"Dev Database","enabled":true,"icon":"","notes":"This secret provides access to the dev database.","names":["DB_HOST","DB_PASS","DB_USER"],"events":["emeekm2ablu"],"categories":[],"plugins":["pmibcla6mg8"],"web_hooks":["example_hook"],"username":"admin","modified":1763880046,"created":1763745419,"revision":2},"type":"plugins","id":"pmibcla6mg8"}]
```

### SQLite.log
SQLite-backed document store operations and lifecycle.

```
[1763880095.555][2025-11-22 22:41:35][joemax.lan][10341][SQLite][debug][9][Cached JSON fetch complete: hosts/smf4j79snhe/data][]
```

### Storage.log
Abstract storage operations across engines (get, put, commit, rollback, etc.).

```
[1763880095.554][2025-11-22 22:41:35][joemax.lan][10341][Storage][transaction][get][users/admin][{"elapsed_ms":0.573}]
```

### Transaction.log
High-level application transactions for auditing and replay (creates, updates, completions, etc.).

```
[1763879911.098][2025-11-22 22:38:31][joemax.lan][10341][Transaction][transaction][server_add][Server connected to the network: raspberrypi (::ffff:10.1.10.92)][{"server_id":"smf4j79snhe","hostname":"raspberrypi","ip":"::ffff:10.1.10.92","groups":["main"],"keywords":["smf4j79snhe"]}]
```

### Unbase.log
Indexing and background record maintenance (e.g., activity index writes).

```
[1763879968.522][2025-11-22 22:39:28][joemax.lan][10341][Unbase][debug][6][Insert complete][{"index":"activity","id":"amibcla76g9"}]
```

### User.log
User authentication events and sessions.

```
[1763879912.703][2025-11-22 22:38:32][joemax.lan][10341][User][transaction][user_login][admin][{"ip":"127.0.0.1","headers":{"host":"local.xyops.io:5523","accept":"*/*","content-type":"application/json","origin":"https://local.xyops.io:5523","sec-fetch-site":"same-origin","sec-fetch-mode":"cors","user-agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.1 Safari/605.1.15","referer":"https://local.xyops.io:5523/","sec-fetch-dest":"empty","content-length":"2","accept-language":"en-US,en;q=0.9","priority":"u=3, i","accept-encoding":"gzip, deflate, br","connection":"keep-alive","ssl":1,"https":1}}]
```

### WebServer.log
HTTP/HTTPS server lifecycle, connections, and request metrics.

```
[1763880108.193][2025-11-22 22:41:48][joemax.lan][10341][WebServer][debug][3][HTTPS server on port 5523 has shut down.][{"address":"::","family":"IPv6","port":5523,"ssl":true}]
```

### Workflow.log
Workflow node execution and job orchestration details.

```
[1763841977.078][2025-11-22 12:06:17][joemax.lan][92938][Workflow][debug][6][Workflow is complete][{"job":"jmiapyuj8at"}]
```

### xyOps.log
Main application lifecycle and component startup/shutdown.

```
[1763880108.203][2025-11-22 22:41:48][joemax.lan][10341][xyOps][debug][2][Shutdown complete, exiting][]
```

## Crash Log

If xyOps crashes or an uncaught exception occurs and [log_crashes](config.md#log_crashes) is `true`, a `crash.log` file is written into [log_dir](config.md#log_dir) containing the most recent crash timestamp and JavaScript stack trace. This file is plain text (not bracketed columns).

Example format (stack trace truncated):

```
2025-11-22 22:41:48 Uncaught exception: Error: Something bad happened
    at Object.<anonymous> (app.js:123:45)
    at Module._compile (node:internal/modules/cjs/loader:1356:14)
    ...
```

## Nightly Archival

xyOps automatically archives server logs every night at midnight (local server time):

- All `.log` files in [log_dir](config.md#log_dir) are compressed with gzip and copied to a date-based path.
- The destination is controlled by [log_archive_path](config.md#log_archive_path), which supports date/time placeholders.

Default path pattern:

```
logs/archives/[yyyy]/[mm]/[dd]/[filename]-[yyyy]-[mm]-[dd].log.gz
```

Where `[filename]` expands to the source filename without the extension (e.g., `API` for `API.log`). The Maintenance component performs this task nightly; see `Maint.log` for status entries.

If you prefer a single combined log (e.g., `xyops.log`), set [log_filename](config.md#log_filename) accordingly; the nightly archival will still archive that single file using the same pattern.
