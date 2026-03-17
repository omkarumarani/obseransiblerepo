# Page Descriptions

This document contains short descriptions of each top-level page.  They are displayed in the UI if the user has the preference enabled.

## Search

Completed jobs are stored for analysis, troubleshooting, and auditing. Job history includes results, timing, parameters, logs, output data, and files, so you can verify behavior and compare outcomes across runs.

The Job Search page lets you browse and filter this history. Narrow by result (success, error, warning, critical, aborted), event or workflow, category, tags, source (scheduler, manual, API key, workflow, action, alert, plugin), plugin, server, group, and date range. You can also search inside logs and attached text files using plain text or regular expressions with optional case sensitivity.

Save frequent queries as presets that appear in the sidebar for quick access. Presets are per‑user so you can keep personal workflows without affecting others. 

Learn more: [Events](events.md), [Workflows](workflows.md), [Tags](tags.md).

## Tickets

**Tickets** are lightweight records for tracking issues, features, releases, changes, maintenance, and questions. They capture a subject, body, assignees and CC lists, optional due dates, tags, and files, and they can link to automation resources including events, jobs, and alerts.

The Ticket Search page helps you find and manage this work. Click "**New Ticket**" to create one, or search by full‑text in the body, type, status (open, closed, draft), assignee, author, category, server, tags, and date range.

Tickets can embed runnable events so you can launch jobs directly from a ticket. Attached files flow into those jobs as inputs and outputs are linked back for traceability. 

Learn more: [Tickets](tickets.md).

## Events

**Events** define how jobs run: what code to execute (plugin and parameters), where to run it (targets and selection algorithm), when to run (triggers), and how to react (actions and limits). Each event launch produces a job with full lifecycle tracking.

The Event List page lets you browse, search, and manage events. Click "**New Event**" to add one, or filter by keywords (title), status (enabled or disabled), category, target (group or server), plugin, tags, triggers, actions, or author. Open any event to view or edit its configuration.

Category defaults merge into event jobs, and per‑event limits and actions can refine behavior. Include a manual trigger if you want on‑demand runs via UI or API. 

Learn more: [Events](events.md), [Categories](categories.md), [Plugins](plugins.md).

## Workflows

**Workflows** are visual graphs that orchestrate multiple jobs with flow control. They connect triggers, event or job nodes, controllers, actions, and limits to express parallelism, branching, fan‑out, and fan‑in.

The Workflow List page shows all workflow events. Click "**New Workflow**" to create a workflow, or filter the existing ones by keywords (title), status (enabled or disabled), category, target (group or server), tags, triggers, actions, or author to find what you need.

Common controllers include split, join, repeat, multiplex, decision, and wait. The editor supports testing with custom inputs and optional action or limit suppression. 

Learn more: [Workflows](workflows.md).

## Categories

**Categories** are user‑defined groupings for events and workflows. They help organize automation, control visibility, and apply default actions and limits to all jobs inside the category.

The Category List page lets you create and manage categories. Set a title, color, icon, notes, enabled flag, and sort order. You can import/export and reorder categories to match your preferred layout.

Defaults from categories merge into job runs, standardizing notifications and resource policies. Disabling a category prevents scheduling and manual runs for everything it contains. 

Learn more: [Categories](categories.md), [Actions](actions.md), [Limits](limits.md).

## Buckets

**Storage Buckets** are durable stores for JSON data and files that jobs and workflows can share. They are useful for cross‑job handoffs, artifact storage, and long‑lived shared state.

The Storage Bucket List page lets you create buckets, edit the JSON data pane, and upload or download files. You can replace or remove objects as needed, with filename normalization and size/count limits enforced by the server.

Jobs can fetch from a bucket at start and store data or files on completion via actions, allowing loose coupling between processes and teams. 

Learn more: [Buckets](buckets.md).

## Tags

**Tags** are reusable labels you can attach to events, jobs, and tickets. They provide quick visual cues, enable powerful searches, and can drive conditional automation.

The Tag List page lets you create, edit, and delete tag definitions with titles, optional icons, and notes. Tag IDs are the reference keys used by actions and plugins.

Actions can be conditioned on tags present at job completion, and plugins can push tags at runtime to annotate jobs. 

Learn more: [Tags](tags.md).

## Alerts

**Alerts** continuously evaluate live server data and fire when conditions are met. Definitions specify scope, expressions, messages, and actions; each firing becomes an invocation with its own lifecycle.

The Alert Search page shows active and historical invocations so you can review conditions, timing, and downstream actions. Filter by alert definition, server, group, and date range to isolate incidents or trends.

On fire or clear, actions can notify channels, create tickets, run events, and take snapshots. Alerts can also block new jobs or abort running jobs on affected servers. 

Learn more: [Alerts](alerts.md).

## Servers

**Servers** are worker nodes that execute jobs and collect metrics. Each runs the xySat agent, maintains a connection to a conductor, and reports host details and monitoring data.

The Server List page shows the active fleet with labels, hostnames, IPs, OS and CPU details, and status. Filter by keywords and platform attributes, use "**Search History**" for offline servers, or click "**Add Server**" to generate a one‑line installer for Docker, Linux, macOS, or Windows.

Opening a server presents live and historical charts, current processes and network connections, running and upcoming jobs, and active alerts with links to details. 

Learn more: [Servers](servers.md).

## Groups

**Server Groups** organize servers into logical sets used for targeting, dashboards, and alert scoping. Membership can be automatic via hostname regex or assigned manually per server.

The Server Group List page lets you create, edit, and reorder groups. Click "**New Group**" to add one, set an optional hostname match, notes, and default alert actions.

Opening a group shows aggregated charts, processes and connections across members, running and upcoming jobs, alerts, and controls for snapshots and watches. 

Learn more: [Server Groups](groups.md).

## Snapshots

**Snapshots** capture a point‑in‑time view of a server or a whole group, including metrics, processes, connections, and context like jobs and alerts. They are ideal for forensics and before/after comparisons.

The Snapshot History page lists captured records. Filter by source (alert, user, watch, job), server, group, and date range to locate relevant entries and trends.

Opening a snapshot shows frozen server or group data with quick metrics and minute‑level monitors, processes, connections, jobs, and alerts from the exact capture moment. 

Learn more: [Snapshots](snapshots.md).

## MyAccount

Your account profile includes your identity, credentials, roles, and avatar. Keeping this up to date helps others recognize you in activity logs and lists.

The My Account page lets you change your display name, email address, and password, and upload or replace your avatar image. You can also select an icon to display alongside your name in the UI.

The page shows which roles you are assigned so you can understand your effective permissions at a glance. 

## MySettings

Preferences control the look, feel, and behavior of the UI for your account. They include localization, accessibility, notifications, media, and keyboard shortcuts.

The My Settings page lets you configure locale, region, timezone, number format, and hour cycle. You can adjust motion, contrast, vision accessibility, notifications, visual effects, and streamer mode, along with volume, brightness, contrast, hue, and saturation.

Keyboard shortcuts can be tailored for efficiency. Settings travel with your account and apply across devices. 

## MySecurity

Your personal security log shows account activity such as logins and sensitive changes. Each entry includes metadata like IP addresses and user agents.

The My Security page allows you to audit this history and confirm that recent actions are legitimate. It is a good habit to review periodically.

Use Logout All Sessions to invalidate every other session while keeping the current one active.

## ActivityLog

The system activity log records creates, updates, deletes, user logins, security actions, and server connectivity changes. It provides an audit trail for administrators.

The Activity Log page presents this history with search and filters so you can isolate actions by user, object type, or time window. Each entry includes helpful metadata for investigations.

Use this page for change reviews, compliance audits, and troubleshooting administrative operations across the platform.

## AlertSetup

Alert definitions describe what to watch on servers and what to do when conditions are met. Each definition includes scope, an expression, a message, and actions for fire and clear.

The Alert Setup page lists and manages these definitions. Click "**New Alert**" to create one, test expressions against live data, and configure warm‑up/cool‑down samples, overlays, and options.

Alerts can notify channels, create tickets, run events, take snapshots, and optionally limit or abort jobs on affected servers while active. 

Learn more: [Alerts](alerts.md).

## APIKeys

**API Keys** are access tokens for programmatic use of the REST API. They are like special user accounts for applications, with assignable privileges and optional role grants.

The API Keys page lets you create and manage keys. Click "**New API Key**" to set a title, description, privileges, roles, and an optional expiration. Keys can be disabled or deleted at any time.

Key values are shown only once and stored as salted SHA‑256 hashes thereafter. Scope keys narrowly and expire them when appropriate. 

Learn more: [API Reference](api.md).

## Channels

**Notification Channels** bundle multiple actions such as emailing users, firing a web hook, launching a remediation event, and showing in‑app notifications with sound. Referencing a channel keeps responses consistent.

The Channels page lets you define and manage these bundles. Click "**New Channel**" to pick users, external emails, a web hook, an optional run‑event, an optional sound, and an optional per‑day cap.

Channels execute their sub‑actions in parallel and record results for auditing. Use icons and concise titles for easy recognition. 

Learn more: [Channels](channels.md).

## Conductors

**Conductors** are the main xyOps scheduler servers that coordinate job launches, data ingestion, storage, and the UI. A cluster can run multiple conductors for redundancy; one is primary at any moment.

The Conductors page shows these servers and indicates which is primary and which are online or offline. This helps confirm redundancy and election status.

Use this view to see server status or restart / shutdown conductors.

Learn more: [Servers](servers.md).

## Monitors

**Monitors** track a single numeric metric per server over time, computed from the live data received each minute. They power the charts on server and group pages and can feed alerts.

The Monitors page lets you define these metrics. Click "**New Monitor**" to provide an expression, data type, optional regex extraction, and delta or rate options. You can scope a monitor to specific groups.

Well‑tuned monitors produce clear visualizations and stable alert conditions for capacity and performance tracking. 

Learn more: [Monitors](monitors.md), [Alerts](alerts.md).

## Plugins

**Plugins** extend xyOps with custom logic written in any language. Event plugins run job code on servers, action plugins run on the conductor in response to job or alert actions, monitor plugins emit metrics, and trigger plugins decide when to run events.

The Plugins page lists installed plugins and lets you add new ones. Click "**New Plugin**" to create and configure parameters, icons, and notes for a plugin type.

Plugins can accept parameters, interact with secrets, push updates during runs, and attach files or data to jobs. 

Learn more: [Plugins](plugins.md).

## Secrets

**Secrets** are encrypted vaults of key/value variables such as tokens and passwords. They are delivered securely to jobs and web hooks at runtime without exposing plaintext at rest.

The Secrets page lets administrators create secrets, define variables, and assign usage to events, categories, plugins, or web hooks. Enable or disable secrets without deleting them, and decrypt values only when necessary.

At runtime, jobs receive secret variables as environment variables and web hooks expand them in templates. Usage is logged and administrator decryption is audited. 

Learn more: [Secrets](secrets.md).

## System

The System status and maintenance page provides a high‑level view of resource usage and offers on‑demand administrative actions. This helps you gauge health and perform routine upkeep.

The page shows process CPU and memory, database memory and disk usage, cache objects and utilization, and DB row counts. It also lists running internal jobs and connected users with IPs, current page, session duration, and ping RTT.

Maintenance actions include importing, exporting, and deleting bulk data, running nightly maintenance, optimizing the database, resetting daily stats, upgrading worker and master servers, and rotating the secret key.

## Users

**Users** are individual accounts that log in to the UI and API. Permissions come from direct privileges and assigned roles, and access can be restricted to specific categories and groups.

The Users page lists accounts and provides tools for creation, editing, suspension, unlocking, and deletion. Click "**New User**" to set identity fields, initial password, privileges, and roles.

Use this area to manage onboarding and ongoing access, including avatars, password changes, and resource restrictions. 

Learn more: [Users and Roles](users.md).

## Roles

**User Roles** bundle privileges and optional resource restrictions so you can grant capabilities consistently. Assign roles to users to avoid managing dozens of privileges per account.

The Roles page lets you create and manage these bundles. Click "**New Role**" to define privileges and optional category or group limits, then enable it for use.

Changes to roles affect assigned users and take effect shortly after saving. Use roles to standardize policy across teams. 

Learn more: [Users and Roles](users.md).

## WebHooks

**Web Hooks** are outbound HTTP requests that integrate jobs and alerts with external systems such as chat, incident tools, or custom endpoints. They are fully customizable and support templating.

The Web Hooks page lets you create definitions with URL, method, headers, optional body, timeouts, retries, redirect handling, daily caps, and TLS options. Click "**New Web Hook**" to get started.

Templates can include job or alert context and reference assigned secrets for tokens or credentials. Executions record timing and success for troubleshooting. 

Learn more: [Web Hooks](webhooks.md).

## Marketplace

The **xyOps Marketplace** allows you to download and install plugins created by PixlCore and the community.  All plugins are checked by PixlCore before publishing to ensure quality and safety, but always use caution when downloading software from the internet.

The marketplace doesn't actually "host" the Plugins -- it merely provides a search mechanism to discover them.  The Plugins themselves are hosted on package repositories like NPM, PyPI or GitHub, and the marketplace links to them directly.

For your privacy, requests never go out to the PixlCore / xyOps servers.  Instead, everything is hosted entirely on GitHub, including the marketplace index.

Learn more: [Marketplace](marketplace.md).
