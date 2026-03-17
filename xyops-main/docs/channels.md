# Channels

## Overview

Notification Channels in xyOps let you bundle multiple notification targets and follow-up actions under a single reusable name. Instead of attaching individual emails, web hooks, or run-event actions everywhere, you reference a channel from your event/workflow or alert action and xyOps executes the channel's configured actions together.

Typical use case: Create a channel named "Severity 1" that emails your on-call team, sends a Slack/web hook, runs a remediation event, and plays an audible alert in the UI for connected users.

This document explains how channels work, where they are used, what they can do, and provides an example configuration.

## Key Points

- Channels are reusable notification/action bundles that you define once and reference from actions.
- Attach a channel via an action of type "Notify Channel" on events/workflows or alerts. See [Actions](actions.md).
- When a channel runs, xyOps can: email users and explicit addresses, fire a web hook, launch an event, and display in-app notifications with an optional sound for selected users.
- Channels can be enabled/disabled, have an optional icon, and support a per-day rate limit to prevent notification floods.
- Channel execution is recorded in job Activity logs (for jobs) or stored on the alert record (for alerts), including sub-action details.

## Where Channels Are Used

- Event/Workflow actions: Add an action type "Notify Channel" to run on job start, or a completion outcome such as error, warning, etc.
- Alert actions: Add an action with type "Notify Channel" to run on alerts firing or clearing.
- Admin UI: Create and manage channels in Admin → Channels.

See [Actions](actions.md) for action conditions and behavior.

## Privileges

Managing channels requires account privileges:

- `create_channels`: Create notification channels.
- `edit_channels`: Edit existing channels.
- `delete_channels`: Delete channels.

See [Privileges](privileges.md) for details on assigning privileges or roles.

## What Channels Can Do

When a channel is invoked, xyOps executes these configured actions in parallel:

- Email: Email all channel users plus any extra addresses provided.
- Web Hook: Fire a configured outbound web hook with a rich, templated payload.
- Run Event: Launch a specific event as a follow-up/remediation.
- In-App Notify: Send a UI notification to all channel users; can optionally play a sound.

Notes:

- Email and web hook payloads are templated based on context. For jobs, payloads include job links, log excerpts, and metadata. For alerts, payloads include server and alert details with direct links.
- Launched events inherit context: job actions include the parent job's output data/files; alert actions include alert metadata in the child job's input data.
- Channel actions are executed in parallel, and their individual results are aggregated into the parent action's details for auditing.
- Disabled channels are skipped. If a referenced channel is disabled, the action records a message and does nothing.

## Rate Limiting

Channels support an optional per-day cap. When the limit is reached, the channel action is skipped for the rest of the day and a message is recorded. Counts reset daily at midnight (local server timezone).

## Channel Object

Channels are first-class objects with these key properties. See the canonical definition in [Channel](data.md#channel).

- `id`: Unique alphanumeric identifier. If omitted on creation, xyOps auto-generates one.
- `title`: Display title shown in the UI and notifications.
- `enabled`: Enable/disable the channel.
- `icon`: Optional Material Design icon name for UI display.
- `users`: Array of usernames to notify. Used for both email lookup and in-app notifications.
- `email`: Optional comma-separated email address list (external recipients).
- `web_hook`: Optional [WebHook.id](data.md#webhook-id) to fire.
- `run_event`: Optional [Event.id](data.md#event-id) to launch.
- `sound`: Optional `.mp3` filename to play for channel users in the UI notification.
- `max_per_day`: Optional cap on channel invocations per day (0 = unlimited).
- `notes`: Optional free-form notes.

## Example Channel

Example JSON for a "Severity 1" channel:

```json
{
  "id": "sev1",
  "title": "Severity 1",
  "enabled": true,
  "icon": "bullhorn-outline",
  "users": ["oncall", "noc"],
  "email": "ops@example.com, sre@example.com",
  "web_hook": "slack_ops",
  "run_event": "auto_remediate",
  "sound": "attention-3.mp3",
  "max_per_day": 100,
  "notes": "Used to page on-call for P1 incidents."
}
```

## Using a Channel in Actions

Attach the channel via an action of type "Notify Channel". Example action snippets (JSON):

- Job action (on error):

```json
{
  "enabled": true,
  "condition": "error",
  "type": "channel",
  "channel_id": "sev1"
}
```

- Alert action (when fired):

```json
{
  "enabled": true,
  "condition": "alert_new",
  "type": "channel",
  "channel_id": "sev1"
}
```

See [Actions → Channel](actions.md#channel) for action semantics and deduplication.

## Behavior Details

- Parallel execution: Email, web hook, run event, and UI notifications execute concurrently.
- Deduplication: xyOps dedupes actions by type and target at the job/alert level. Multiple references to the same channel within a single trigger run only once; their contained actions execute once as part of the channel.
- In-app notifications: All `users` receive a popup notification with an optional sound. Job actions link to the job; alert actions link to the alert.
- Templates: Email/web hook messages use standard job/alert templates. Channel configuration does not add custom text; customize per-action text by using direct email/web hook actions if needed.
- Auditing: For jobs, sub-action results are aggregated into the job's Activity log under the channel action. For alerts, results are stored with the alert's action history.

## Managing Channels

- Create/Edit/Delete in Admin → Channels.
- Toggle enabled status from the channel list or editor.
- Choose an optional icon and sound. Sound files must be `.mp3`. The UI provides a curated list; you can preview sounds in the editor.
- Set `max_per_day` to cap total channel invocations per day.

## API Endpoints

For automation and tools, these endpoints manage channels:

- `app/get_channels/v1`: List all channels.
- `app/get_channel/v1`: Fetch a single channel by `id`.
- `app/create_channel/v1`: Create a channel. If `id` is omitted, one is auto-generated.
- `app/update_channel/v1`: Update a channel.
- `app/delete_channel/v1`: Delete a channel.

See [Channels API](api.md#channels) for more API details.

## Notes and Tips

- Keep channels focused: Create separate channels for different severities or teams (e.g., `ops_oncall`, `security_incidents`, `customer_success`).
- Prefer channels for consistency: Reference the same channel across multiple events and alerts to keep your notification patterns uniform.
- Use run-event carefully: Ensure the remediation event is idempotent and safe to trigger on repeated conditions.
- Rate limits are per channel: If you need separate caps for different targets, split into multiple channels.

## See Also

- API: [Channels](api.md#channels)
- Actions: [Actions](actions.md)
- Data structures: [Channel](data.md#channel)
- Privileges: [Privileges](privileges.md)
- Web Hooks: [Web Hooks](webhooks.md)
- Events: [Events](events.md)
