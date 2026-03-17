# System Hooks

## Overview

System Hooks let you run custom background actions in response to global activity across xyOps. Activity includes user transactions (create/update/delete of objects, secret access), system events (servers connecting/disconnecting, conductor changes), and general notices (`notice`, `warning`, `error`, `critical`). Hook actions can fire a web hook, run a shell command, send email, and/or create a ticket.

System Hooks are configured in `config.json` under the `hooks` object. Hook actions run in parallel and always in the background.

## Configuration

Hooks are configured in the main `config.json` file (see `sample_conf/config.json`). By default, no hooks are defined:

```json
"hooks": {}
```

Each key under `hooks` is an activity type ID, and the value is an object describing one or more hook actions to run. You can combine actions within the same hook, and they will all fire.

Example configuration:

```json
"hooks": {
	"error": {
		"shell_exec": "/opt/system/bin/notify-error.sh"
	},
	"critical": {
		"shell_exec": "/opt/system/bin/notify-critical.sh",
		"email": "oncall-pager@mycompany.com",
		"ticket": {
			"type": "issue",
			"assignees": ["admin"]
		}
	},
	"secret_access": {
		"web_hook": "mkd0t84o7hwasu0g"
	}
}
```

## Activity Type IDs

- See [Activity.action](data.md#activity-action) for a list of all the activity type IDs you can attach global system hook actions to:
- See [Activity](data.md#activity) for details on the activity payload format and common properties.

## Job and Alert Conditions

All job actions also fire System Hooks, using a `job_` prefix on the action condition. Common examples include:

- `job_complete`: Any job completion (success or failure)
- `job_success`: Successful completion
- `job_error`: Any failed completion
- `job_warning`: Completion with warning code
- `job_critical`: Completion with critical code
- `job_abort`: Job aborted by user or failure condition

Tag-based job actions also emit hooks using the same `job_` prefix (for example, `job_tag:deploy`). See the "Action Conditions" section in [Actions](actions.md) for the full list of job and alert action conditions.

Alert actions also fire System Hooks:

- `alert_new`
- `alert_cleared`

## Firehose Hook

You can configure a catch-all hook using a `*` (asterisk) key. This fires for every activity type:

```json
"hooks": {
	"*": {
		"shell_exec": "/opt/system/bin/firehose.sh"
	}
}
```

## Hook Action Types

System Hooks support these action types. You can use any combination within a single hook, and all actions run in parallel.

### Web Hook

You have two options for firing web hooks:

1. **Simple URL**: Specify a URL directly in `config.json`, and xyOps posts the activity payload as JSON.
2. **Configured web hook**: Reference a web hook ID that is managed inside xyOps for full control over method, headers, body, and templating.

Simple URL example:

```json
"hooks": {
	"alert_new": {
		"url": "http://alerts.mycompany.com/api/new-alert/v1"
	}
}
```

Configured web hook example:

```json
"hooks": {
	"alert_new": {
		"web_hook": "wmkd2yx4yw4ihh7lu"
	},
	"job_error": {
		"web_hook": "wmkd312hugd31hqdh"
	}
}
```

For configured web hooks, see [Web Hooks](webhooks.md). Payloads are based on the [Activity](data.md#activity) format.

### Shell Exec

The `shell_exec` action spawns a process and sends the activity payload to the command's STDIN. The payload is JSON on a single line (i.e. [xyOps Wire Protocol](xywp.md)).

Example:

```json
"hooks": {
	"critical": {
		"shell_exec": "/path/to/script.sh"
	}
}
```

The command always runs on the primary conductor server, so please use with extreme caution.  To debug these, grep for `System Shell Hook` in the the `/opt/xyops/logs/Action.log` log.  Increase your global [debug_level](config.md#debug_level) to `9` for more verbose logging.

### Send Email

The `email` action sends a generic activity summary email. It includes the activity summary, full payload (JSON), and any details if provided (common for `notice`, `warning`, `error`, `critical`). Multiple recipients may be specified as a comma-separated list.

Example:

```json
"hooks": {
	"critical": {
		"email": "oncall-pager@mycompany.com"
	}
}
```

Note that to send emails for server alerts (a very common configuration) it is **much** better to use the [alert_universal_actions](config.md#alert_universal_actions) feature, which generates a proper alert-centric email (emails generated from system hooks are more generic).  Recommended configuration for global alert emails:

```json
"alert_universal_actions": [
	{
		"enabled": true,
		"hidden": true,
		"condition": "alert_new",
		"type": "snapshot"
	},
	{
		"enabled": true,
		"condition": "alert_new",
		"type": "email",
		"email": "oncall-pager@mycompany.com"
	}
]
```

The same goes for job failures.  If you want an email for all job failures, use the [job_universal_actions](config.md#job_universal_actions) feature instead, as it generates a more proper email for the specific action:

```json
"job_universal_actions": {
	"default": [
		{
			"enabled": true,
			"hidden": false,
			"condition": "error",
			"type": "snapshot"
		},
		{
			"enabled": true,
			"condition": "error",
			"type": "email",
			"email": "oncall-pager@mycompany.com"
		}
	],
	"workflow": [
		{
			"enabled": true,
			"condition": "error",
			"type": "email",
			"email": "oncall-pager@mycompany.com"
		}
	]
}
```

### Create Ticket

The `ticket` action creates a ticket for the activity. Any ticket fields may be included here. See [Tickets](tickets.md) and the [Ticket](data.md#ticket) definition for field details.

Example:

```json
"hooks": {
	"critical": {
		"ticket": {
			"assignees": ["admin"],
			"type": "issue"
		}
	}
}
```

Note that assigned tickets automatically send out an email to all assignees (assuming [tickets.email_enabled](config.md#tickets-email_enabled) is enabled), so this is yet another way to receive an email for specific activity reports.

If you want to create an extremely loud and annoying ticket, set the `due` property to `today`, which will start sending out daily overdue reminders to all assignees:

```json
"hooks": {
	"critical": {
		"ticket": {
			"assignees": ["admin"],
			"type": "issue",
			"due": "today"
		}
	}
}
```

## Example Payloads

These are sample activity JSON payloads sent to shell exec and simple web hooks.  Note that in practice they are all compacted onto a single line.

Alert Fired (`alert_new`):

```json
{
	"xy": 1,
	"alert": {
		"date": 1768336814,
		"exp": "memory.available < 1073741824",
		"message": "Server has low free memory: 906.2 MB",
		"count": 1,
		"notified": true,
		"id": "amkd22yuuryqbzol",
		"active": true,
		"alert": "amkd22ytorq8q9k3",
		"server": "saturn1",
		"groups": [
			"main",
			"prod"
		],
		"modified": 1768336814
	},
	"description": "New alert for: saturn1: Server has low free memory: 906.2 MB",
	"action": "alert_new",
	"epoch": 1768336814,
	"id": "amkd22yuurzr9emy",
	"keywords": [],
	"text": "New alert for: saturn1: Server has low free memory: 906.2 MB",
	"message": "New alert for: saturn1: Server has low free memory: 906.2 MB",
	"content": "New alert for: saturn1: Server has low free memory: 906.2 MB"
}
```

Critical Error (`critical`):

```json
{
	"xy": 1,
	"description": "Crash log was found on worker server.",
	"details": "**Log Contents:**\\n\\n```\\n(Actual crash log here)\n```\n",
	"server": "api-prod-01",
	"action": "critical",
	"epoch": 1768283576,
	"id": "amkc6dvvd1nci7tl",
	"keywords": [],
	"text": "Crash log was found on worker server.",
	"message": "Crash log was found on worker server.",
	"content": "Crash log was found on worker server."
}
```

Secret Access (`secret_access`), meaning a user or API key decrypted a secret vault:

```json
{
	"xy": 1,
	"secret": {
		"title": "Dev AI Creds",
		"enabled": false,
		"icon": "",
		"notes": "updated by joe",
		"plugins": [
			"shellplug"
		],
		"categories": [],
		"events": [],
		"web_hooks": [
			"example_hook"
		],
		"id": "zmkd22dfakpie2ns",
		"username": "testuser",
		"modified": 1768336787,
		"created": 1768336787,
		"revision": 3,
		"names": [
			"API_KEY"
		]
	},
	"keywords": [
		"zmkd22dfakpie2ns",
		"testuser",
		"::1"
	],
	"ip": "::1",
	"ips": [
		"::1"
	],
	"headers": {
		"content-type": "application/json",
		"accept-encoding": "gzip, deflate, br",
		"user-agent": "xyOps Unit Tester",
		"content-length": "26",
		"host": "localhost:6622",
		"connection": "keep-alive"
	},
	"username": "testuser",
	"description": "Secret was accessed: Dev AI Creds (zmkd22dfakpie2ns)",
	"action": "secret_access",
	"epoch": 1768336787,
	"id": "amkd22dlulih6e0r",
	"text": "Secret was accessed: Dev AI Creds (zmkd22dfakpie2ns)",
	"message": "Secret was accessed: Dev AI Creds (zmkd22dfakpie2ns)",
	"content": "Secret was accessed: Dev AI Creds (zmkd22dfakpie2ns)"
}
```

State Update (`state_update`), meaning a user or API key toggled the main scheduler master switch:

```json
{
	"xy": 1,
	"key": "scheduler.enabled",
	"value": false,
	"ip": "::1",
	"ips": [
		"::1"
	],
	"headers": {
		"content-type": "application/json",
		"accept-encoding": "gzip, deflate, br",
		"user-agent": "xyOps Unit Tester",
		"content-length": "48",
		"host": "localhost:6622",
		"connection": "keep-alive"
	},
	"username": "testuser",
	"description": "Internal state updated: scheduler.enabled",
	"action": "state_update",
	"epoch": 1768336817,
	"id": "amkd230tlujovg4l",
	"keywords": [
		"testuser",
		"::1"
	],
	"text": "Internal state updated: scheduler.enabled",
	"message": "Internal state updated: scheduler.enabled",
	"content": "Internal state updated: scheduler.enabled"
}
```

Master Primary (`master_primary`), meaning a conductor server has taken over operations as primary:

```json
{
	"xy": 1,
	"host": "localhost",
	"description": "Conductor server is now primary: localhost",
	"action": "master_primary",
	"epoch": 1768336785,
	"id": "amkd22c1fbgcilb1",
	"keywords": [],
	"text": "Conductor server is now primary: localhost",
	"message": "Conductor server is now primary: localhost",
	"content": "Conductor server is now primary: localhost"
}
```

## Tips

- Send email on `critical` to page on-call and include the activity details.
- Use a configured web hook to notify incident systems like OpsGenie on `error` or `critical`.
- Use a firehose hook to stream all activity into a log pipeline via `shell_exec`.
