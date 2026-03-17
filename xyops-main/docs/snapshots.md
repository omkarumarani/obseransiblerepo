# Snapshots

## Overview

Snapshots capture a point-in-time view of everything happening on one server (or across a server group). They're designed for fast forensics, side-by-side comparisons (before/after a deploy, during an incident), and long-term audit trails.

This page explains what snapshots are, what they contain, how to create them (manually or automatically), how watches work, and a few tips for using them effectively.

## Key Points

- A snapshot records the current state of a server (processes, connections, mounts, devices, metrics, jobs, alerts, and more).
- Group snapshots record a whole group at once (all current members, plus recently offline servers), enabling fleet-level forensics.
- Snapshots can be created manually in the UI, by API, or automatically via Actions (on jobs/alerts) and Watches (every minute for a duration).
- Snapshots are visible on the Snapshots page and are linked from servers, groups, jobs and alerts.
- Snapshots are retained up to a global cap (default 100,000) and are pruned nightly. See [Servers → Snapshots and Watches](servers.md#snapshots-and-watches).

## What a Snapshot Contains

All server snapshots include a record with the following:

- Minute sample: A full copy of the current [ServerMonitorData](data.md#servermonitordata), which includes:
  - CPU, memory, load, OS/platform/release/arch, uptime.
  - Full process list and process stats.
  - Active network connections (including listeners).
  - Network interfaces and stats; disk mounts and filesystem stats.
  - Monitors (computed values) and deltas; raw plugin command output.
- Quick metrics: The last 60 seconds of per-second "quick" samples (`quickmon`) for CPU/mem/disk/net ([QuickmonData](data.md#quickmondata)).
- Context: IDs of active jobs and active alerts at capture time. For workflow sub-jobs, parents may be included for context.

Group snapshots add fleet context:

- All current members (online) plus recently offline servers (within the last hour), labeled with online/offline state.
- Per-server [ServerMonitorData](data.md#servermonitordata) objects aligned 1:1 with `servers`.
- Per-server 60-second quick samples aligned 1:1 with `servers`.
- Active alerts and jobs relevant to any member server at capture time.

See the full object shapes in [Data → Snapshot](data.md#snapshot) and [Data → GroupSnapshot](data.md#groupsnapshot).

## Creating Snapshots

You can create snapshots in several ways:

- **Manually (UI)**
  - Server: Open a server page and click Snapshot".
  - Group: Open a group page and click "Snapshot".
- **Automatically via Actions**
  - Add a Snapshot action to a job or alert (see [Actions](actions.md)).
  - Jobs: The job must target a specific server; the snapshot is taken on that server.
  - Alerts: The snapshot is taken on the alert's server when the action triggers.
- **By API**
  - Server: `create_snapshot` -- see [API → create_snapshot](api.md#create_snapshot).
  - Group: `create_group_snapshot` -- see [API → create_group_snapshot](api.md#create_group_snapshot).

Permissions: Creating snapshots (UI or API) requires the [create_snapshots](privileges.md#create_snapshots) privilege.

## Watches

Watches instruct xyOps to take snapshots every minute for a specified duration. Use these to capture short-lived issues or observe changes during a rollout.

- **Server Watch**
  - Set from a server page (UI) or API: [watch_server](api.md#watch_server).
  - Snapshots are taken when that server's minute data arrives (each server's minute offset is deterministically staggered across the fleet).
  - Cancel by setting duration to `0` (UI or API). The UI defaults to 5 minutes.
- **Group Watch**
  - Set from a group page (UI) or API: [watch_group](api.md#watch_group).
  - Snapshots run once per minute on the :30 second mark, capturing all matching servers using their most recent minute samples.
  - Recently offline servers (within the last hour) are included and marked offline.

Notes:

- Staggering: Minute collections are staggered across servers to spread load; server watch snapshot times will reflect each server's offset.
- Provenance: Automatically created snapshots record `source` as `watch`; manually created ones record `source` as `user` and include `username`.

## Viewing and Searching

- UI: Click on "Snapshots" in the sidebar; snapshots also link from server and group pages, and from job/alert activity when actions create them.
- API search: Use [search_snapshots](api.md#search_snapshots) to filter and paginate snapshot history.

## Troubleshooting and Tips

- Prefer watches for transient issues: If a problem is bursty or short-lived, start a short watch (e.g., 5-10 minutes) rather than taking a single manual snapshot.
- Align timing with events: For pre/post comparisons, take one before and one after your change; record links in the related ticket or job notes.
- Troublesome job?  Assign snapshot actions on both job start *and* job complete, to compare the server differences.
- Understand minute vs. second data: The core state is minute-granularity [ServerMonitorData]; the `quickmon` buffer adds the previous 60 seconds of per-second context.
- Group snapshots timing: Group watch runs on :30; servers submit minute samples on staggered offsets. Group snapshots use the latest saved minute for each server.
- Recently offline hosts: Group snapshots include recently offline hosts (last hour) and mark them offline so you still see their last known state.
- Permissions: If you don't see snapshot controls or API calls fail, ensure your user or API Key has [create_snapshots](privileges.md#create_snapshots).

## Learn More

- Data Objects: [Snapshot](data.md#snapshot), [GroupSnapshot](data.md#groupsnapshot), [ServerMonitorData](data.md#servermonitordata), [QuickmonData](data.md#quickmondata)
- API Calls: [create_snapshot](api.md#create_snapshot), [watch_server](api.md#watch_server), [create_group_snapshot](api.md#create_group_snapshot), [watch_group](api.md#watch_group), [search_snapshots](api.md#search_snapshots)
- See Also: [Servers](servers.md), [Groups](groups.md), [Actions](actions.md)
