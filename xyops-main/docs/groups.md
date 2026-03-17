# Server Groups

## Overview

Server Groups (often simply called "Groups") let you organize multiple servers into logical sets and operate on them as a unit. A single server may belong to any number of groups. Groups power event targeting (run jobs against a group rather than a specific host), aggregate live and historical views in the UI, provide default alert actions, and enable group snapshots and watches.

This document explains what groups are, how servers join groups (manually and automatically), how events target groups and choose servers, how default alert actions work, how to take group snapshots and watches, and what you can do on the group UI pages.

## Key Points

- Groups are named collections of servers; a server can be in multiple groups.
- Servers can join groups automatically via a hostname regular expression match, or you can assign groups manually on each server.
- Events can target groups for running jobs; xyOps selects one eligible server from the group per job using a configurable algorithm.
- Each group has a dedicated UI page showing the group's servers, live status, monitors, processes, connections, jobs, upcoming jobs, and stats.
- Groups can define default alert actions that run when any alert fires or clears on any server in the group.
- You can take a snapshot of a group on demand, or set a "watch" to take snapshots every minute for a duration.

## Creating and Editing Groups

- **Location**: Sidebar → Monitoring → Groups.
- **Create**: Click "New Group...", set a title, optional icon, and an optional hostname regex for auto-assignment. You can also add default alert actions and notes.
- **Edit**: Click a group → Edit Group to change fields; ID is immutable.
- **Reorder**: Drag rows in the list to change sort order.
- **Import/Export**: Use the Import/Export buttons on the list and editor.
- **Permissions**: Creating, editing, deleting groups require privileges [create_groups](privileges.md#create_groups), [edit_groups](privileges.md#edit_groups), and [delete_groups](privileges.md#delete_groups) respectively.

Group fields:

- **Title**: Display name for the group.
- **Icon**: Optional Material Design Icon ID.
- **Hostname Match**: Regular expression to auto-assign servers by hostname. Leave blank to match none; use `.+` to match all servers.
- **Alert Actions**: Default alert actions for alerts in this group (details below).
- **Notes**: Optional freeform text.

## Adding Servers to Groups

Servers can be added to groups in two ways, and a server may belong to multiple groups at once.

1. **Automatic by Hostname**
	- Each group may specify a hostname regular expression in `hostname_match`.
	- When a server connects (or its hostname changes), xyOps tests it against all groups and assigns any matching groups.
	- This automatic assignment is re-evaluated whenever groups change (create/edit/delete) and propagated to servers and storage.
	- To match all servers, use `.+`.
2. **Manual Assignment per Server**
	- From any Server page, choose "Edit Server..." and set Groups explicitly, or use "Add Server..." on a Group view.
	- When you manually assign groups to a server, the server's "Auto Group" behavior is disabled for that server.
	- To return a server to automatic assignment, clear its manual group list and re-enable Auto Group in the server editor.

Notes:

- Group membership is additive: a server can be in many groups via both automatic and manual assignment (unless Auto Group is disabled by manual override).
- Event targeting honors the server's current live groups; changes take effect immediately for new jobs.

## Default Alert Actions

Groups can define default alert actions that run when any alert fires or clears on any server in the group.

- In the Group editor, under "Alert Actions".
- When they run: On `alert_new` (alert fired) and/or `alert_cleared` (alert cleared), based on each action's `condition`.
- How they combine: For a given alert event, the final action list is a merge of:
	- The alert definition's own actions.
	- All matching servers' group default actions (for each group the server is in).
	- Universal alert actions from configuration.
	- Actions are deduplicated by type + target (e.g., same email recipients, same web hook ID).
- Target context: Group actions include group and server context in notifications (emails/web hooks include group titles and links).

See [Actions](actions.md) for supported types and parameters.

## Targeting Events at Groups

Instead of selecting specific servers, you can target one or more Groups for an Event. At job launch time, xyOps resolves groups to the set of currently online, enabled servers and selects one server for the job using the event's selection algorithm.

Behavior:

- **Resolution**: Event `targets` may include server IDs and/or group IDs. Groups expand to their member servers (as of launch time).
- **Eligibility**: Only online and enabled servers are considered. Servers may be excluded if certain active alerts specify "limit jobs" (alert-level setting).
- **No servers available**: If none qualify, xyOps can queue the job if the event defines a Queue limit; otherwise it fails immediately. See [Limits](limits.md#max-queue-limit).

Selection algorithms:

- `random`: Pick a random server from the eligible set.
- `round_robin`: Cycle through the eligible set per event; persists across runs.
- `prefer_first`: Pick the first server after sorting by hostname ascending.
- `prefer_last`: Pick the last server after sorting by hostname ascending.
- `least_cpu`: Pick the server with the lowest current average CPU load (`info.cpu.avgLoad`).
- `least_mem`: Pick the server with the lowest current active memory usage (`info.memory.active`).
- `monitor:<id>`: Pick the server with the lowest value of the specified monitor.

Details:

- The eligible set is de-duplicated and built from all targets (groups and servers), then filtered by online/enabled status and alert exclusion.
- For `round_robin`, xyOps maintains per-event rotation state so distribution is fair across restarts or conductor failover.

See [Events → Server Selection](events.md#server-selection) and [Data → Event.algo](data.md#event-algo) for additional context.

## Group Snapshots and Watches

Snapshots capture the group's state at a point in time (including member servers, recent metrics, jobs, and alerts). Watches automate snapshots every minute for a duration.

- **Take snapshot**: On a Group view, click "Snapshot". Requires privilege [create_snapshots](privileges.md#create_snapshots).
- **Group watch**: Click the Watch button, set a duration, and xyOps will take a group snapshot every minute until the duration elapses. Set duration to 0 to cancel.
- **History**: Group snapshots appear on the Snapshots page and link back to the group.
- **Contents**: A Group snapshot includes the group definition, the matching servers, each server's last minute of "quick" metrics, active jobs on those servers, and alerts touching the group.

See also: [Snapshots](snapshots.md) and [Data → GroupSnapshot](data.md#groupsnapshot).

## Group UI

Each group has a dedicated live view aggregating all member servers. From Monitoring → Groups, click a group to open its page.

What you'll see:

- Summary: Group ID/title/icon, hostname match regex, server count, alert action count, author, created/modified, and fleet breakdown (architectures, OSes, CPU types, virtualization).
- Controls: Edit Group, Snapshot, Watch, Add Server, plus shortcuts to Group History, Alert History, and Job History.
- Server table: All servers in the group (online and recently offline), with live status, resource donuts, running jobs, and controls. Supports filtering and selection.
- Quick Look: Live per-second charts for CPU, memory, disk, and network across visible servers for the last minute.
- Memory and CPU details: Aggregated memory/CPU breakdowns per server with merging options for compact views.
- Monitors: Last hour of configured monitors, with filter and chart sizing; one layer per server.
- Processes and Connections: Aggregated current process table and active network connections across the group with filters.
- Jobs: Active jobs running on any server in the group; includes progress bars and remaining time.
- Upcoming jobs: Predicted future jobs that will land on any server in the group (based on event schedules targeting the group).
- Alerts: Active alerts affecting any member server, with links to detail.

## API

The following endpoints back groups in the UI and automation:

- List groups: `app/get_groups`
- Get group: `app/get_group`
- Create group: `app/create_group`
- Update group: `app/update_group`
- Delete group: `app/delete_group`
- Reorder groups: `app/multi_update_group`
- Create group snapshot: `app/create_group_snapshot`
- Set/cancel watch: `app/watch_group`

See [API](api.md) for request/response details.

## Best Practices

- Prefer automatic grouping with clear, stable hostname patterns (e.g., environment or role prefixes). Fall back to manual assignments for one-offs.
- Use multiple groups to express orthogonal concepts (e.g., "prod", "db", "east"), then target events with multiple groups as needed.
- Combine group default alert actions with alert-specific actions to keep on-call routing centralized.
- For heterogeneous fleets, consider `monitor:<id>` selection to route jobs based on current load or capacity.
