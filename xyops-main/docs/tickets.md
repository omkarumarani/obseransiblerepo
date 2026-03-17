# Tickets

## Overview

Tickets in xyOps provide a lightweight, integrated way to track issues, releases, changes, incidents, and any operational work that benefits from an audit trail, comments, files, and automation. Tickets live alongside jobs, alerts, servers, and workflows, and can both react to the system (auto-created from jobs or alerts) and drive the system (run events/jobs directly from a ticket).


## Properties

Tickets are simple JSON records with a few core and optional fields. The full schema is documented at [Ticket](data.md#ticket). Key properties are summarized here.

- `subject`: Short summary/title. HTML is stripped.
- `body`: Markdown content for the ticket. Useful for runbooks, context from jobs/alerts, and checklists.
- `type`: One of issue, feature, release, change, maintenance, question, other. Controls presentation only; choose what best fits your workflow.
- `status`: One of draft, open, closed.
  - `draft`: Suppresses all email notifications. Use for drafting without notifying anyone.
  - `open`: Normal active ticket state.
  - `closed`: Completed/resolved. Closing is recorded in change history and can be searched.
- `assignees`: Array of usernames responsible for the ticket. Receive update emails and overdue notices.
- `cc`: Array of usernames to also receive update emails (no overdue notices).
- `notify`: Array of custom email addresses for updates (no overdue notices). Useful for team lists like `ops-team@company.com`.
- `category`: Optional [Category.id](data.md#category-id). Auto-set when tickets are created from jobs to match the job's category.
- `tags`: Array of [Tag.id](data.md#tag-id)s. Auto-set from the source job's tags when auto-created.
- `server`: Optional [Server.id](data.md#server-id). Auto-set when created from jobs/alerts that reference a server.
- `due`: Optional due date (Unix seconds). After the date passes, daily overdue notices are emailed to assignees.
- `files`: Array of uploaded files attached to the ticket. Files are listed on the ticket page and passed as inputs to jobs launched from the ticket's events.
- `events`: A list of event stubs that can run jobs from the ticket. Each event may override targets, selection algorithm, tags, and parameter defaults.
- `changes`: The change and comment history. Includes structured "change" entries and "comment" entries.


## Creating Tickets

You can create tickets manually, through the API, and automatically via job/alert actions.

### Manually

- Click "New Ticket" in the sidebar.
- Fill subject, body (Markdown), type, status, category, server, assignees, cc, notify, tags, and due.
- Attach files if needed (these appear under Ticket Files and will be passed to jobs launched from the ticket).
- Save as Draft to suppress notifications until ready.

### API

Use [create_ticket](api.md#create_ticket) to create tickets programmatically. You may post JSON or multipart/form-data (to upload files). See [API](api.md) for full details and examples.

- JSON: `POST /api/app/create_ticket/v1` with the ticket fields.
- File upload: Use `multipart/form-data` with a `json` field containing the ticket JSON string, plus one or more file fields to attach.

Related APIs:

- [update_ticket](api.md#update_ticket): Shallow-merge updates; server detects and records changes.
- [add_ticket_change](api.md#add_ticket_change) and [update_ticket_change](api.md#update_ticket_change): Add/edit/delete comments or change entries.
- [upload_user_ticket_files](api.md#upload_user_ticket_files): Upload and attach files to ticket.
- [delete_ticket_file](api.md#delete_ticket_file): Remove an attached file.
- [delete_ticket](api.md#delete_ticket): Permanently delete a ticket.
- [search_tickets](api.md#search_tickets): Search with pagination and sorting; supports compact mode for grids.

### Job Action

Jobs can create tickets on start or completion based on outcome or tags. Add a "Create Ticket" action to an event, workflow node, or via category/universal defaults. When fired:

- The ticket body is auto-generated (template: job) with useful context (job details, performance, log excerpt, links).
- Category, tags, and server fields are auto-populated from the job when applicable.
- The new ticket is added to the originating job for traceability.

See [Actions](actions.md) for action configuration.

### Alert Actions

Alerts can create tickets when an alert fires (or clears, if desired). Add a "Create Ticket" alert action. When fired:

- The ticket body is auto-generated (template: alert) with server and alert context, links to the alert and server, and optionally active job summaries.
- Server is populated from the firing server; tags can be set from the action.
- The new ticket is added to the alert invocation record.


## Ticket Events

Ticket events attach runnable events (jobs) to a ticket with optional parameter overrides.  Just click the "Add Event" button and select an event.

From the ticket view you can run any attached event. When a job is launched from a ticket:

- The ticket is associated to the job.
- Any ticket files are passed as job input files.
- Files produced by previous ticket-launched jobs can also be chained into subsequent runs from the ticket view.

This makes tickets a powerful control plane for CI/CD: create a ticket for a release, attach deploy/test/rollback events, upload artifacts to the ticket, then run jobs from the ticket and keep the entire history centralized.


## Ticket Files

- Attach files in the UI or upload via API when creating a ticket. Files are stored server-side and listed on the ticket.
- Files attached to a ticket are automatically provided to jobs launched from the ticket's events.
- Files can be removed from the ticket; deletion removes the record and the stored object.
- File expiration is governed by configuration (see [file_expiration](config.md#file_expiration)). Expired files are cleaned up automatically.


## Comments and Changes

Any user with the [edit_tickets](privileges.md#edit_tickets) privilege can add comments. Comments are stored in changes with type comment, and support Markdown formatting.

- **Change history**: Edits to key fields (subject, status, type, category, server, assignees, due, cc, notify, tags) are recorded as structured change entries. Draft tickets do not record changes or send notifications.
- **Email updates**: Assignees, cc users, and notify emails receive batched email updates (debounced) with a summary of changes and any new comments. If the body text was changed, the full body is included in the update.
- **Overdue notices**: After due date has passed, daily overdue emails are sent to assignees only.


## Searching

To perform a ticket search, click on ticket search preset links in the sidebar, e.g. "**All Tickets**".  Then, enter one or more words into the search field, and hit enter (or click the **Search** button).  By default the ticket subject and body text are searched.  Searches are not case-sensitive.  If you enter multiple words, they all must be found in the ticket for it to be included in the search results.  However, the words don't necessarily have to appear in order.  For example, consider this search query:

```
zip targeting
```

This would find any tickets that contained both words (`zip` and `targeting`), but they do not have to appear next to together.  They each can be anywhere in the ticket.  If you want to match an exact phrase, surround it with "double-quotes", like this:

```
"zip targeting"
```

This would only show messages that contained the *exact phrase* in quotes, i.e. the two words in sequence.

### Negative Matches

You can augment a search so that it *excludes* certain words or phrases.  To do this, prefix the negative words or phrases with a hyphen (`-`).  Example:

```
"zip targeting" -birds -cats -frogs
```

This would find messages that contain the exact phrase "zip targeting", but **not** the words `birds`, `cats` or `frogs`.  Note that negative words can only take away from an existing search result, so you need to start with some positive (normal matching) words.

### OR Matches

To find messages that contain any of a set of words (known as an "OR" match), separate them by pipe (`|`) characters.  Example:

```
campaign | memory | performance
```

This would find tickets that contain **any** of the specified words.  Note that you cannot combine an OR and an AND search in the same search query, so it's either one or the other (unless you use [PxQL](https://github.com/jhuckaby/pixl-server-storage/blob/master/docs/Indexer.md#pxql-queries)).

### Search Options

You can search more than just the ticket text.  Click the "**Options**" button to reveal several drop-down menus to narrow your search, using criteria such as type, assignee, tags, date range, and more.  You can also use a GitHub-style syntax to customize your search inside the query text box, by specifying one or more field IDs followed by a colon.  Here is an example:

```
subject:yum status:open created:>=2020-01-01
```

This would search for all open tickets that contain the word "yum" specifically in their subject line, and were created in or after the year 2020.  Here is a list of all the fields you can use to narrow your search:

| Field ID | Description |
|----------|-------------|
| `subject` | Search the ticket subject line, e.g. `subject:yum`. |
| `body` | Search the ticket body text (which also includes the subject), e.g. `body:memory usage` (this is the default search field if none are specified). |
| `changes` | Search the ticket changes (i.e. comments and field changes), e.g. `changes:rolled`. |
| `status` | Search the ticket status field, e.g. `status:open` or `status:closed`. |
| `username` | Search the ticket author, e.g. `username:admin`. |
| `assignees` | Search the ticket assignees, e.g. `assignees:admin`. |
| `cc` | Search the ticket carbon-copy list, e.g. `cc:admin`. |
| `type` | Search the ticket type ID, e.g. `type:issue`. |
| `category` | Search the ticket category, e.g. `category:prod`. |
| `tags` | Search the ticket tags, e.g. `tags:important`. |
| `created` | Search the ticket creation date, e.g. `created:>2020-02-01`.  This is a date field which supports ranges.  See [Simple Queries](https://github.com/jhuckaby/pixl-server-storage/blob/master/docs/Indexer.md#simple-queries) for details. |
| `due` | Search the ticket due date, e.g. `due:<today`.  This is a date field which supports ranges.  See [Simple Queries](https://github.com/jhuckaby/pixl-server-storage/blob/master/docs/Indexer.md#simple-queries) for details. |
| `num` | Search the ticket number, e.g. `num:>5000`.  This is an integer number field which supports ranges.  See [Simple Queries](https://github.com/jhuckaby/pixl-server-storage/blob/master/docs/Indexer.md#simple-queries) for details. |

If you specify multiple fields, they are matched using logical AND.  You can, however, specify multiple values inside each field using a pipe (|) character for an inner logical OR match.  Example:

```
status:open|closed tags:important
```

If you need more complex queries, consider using [PxQL](https://github.com/jhuckaby/pixl-server-storage/blob/master/docs/Indexer.md#pxql-queries) syntax, which is fully supported.

### Search Examples

Here are a few search query examples you might find useful:

| Name | Query |
|------|-------|
| All tickets | `*` |
| All issues | `type:issue` |
| All non-important | `num:>0 tags:-important` |
| All overdue tickets | `status:open due:<today` |
| Created in date range | `created:2021/02/01..2021/03/01` |
| Open for N days | `status:open created:<2021/02/01` |
| Open and important | `status:open tags:important` |
| Ticket number range | `num:>=5000 num:<6000` |
| Search only subjects | `subject:zip targeting` |
| Search only comments | `changes:thank you` |

### Search Presets

Once you have your search working how you want it, you can "save" it to your account as a search preset.  Saved presets will appear in the sidebar under "**Ticket Searches**", so you can get back to them with one single click, and see updated results.

To save a search query, click on the "**Save Preset**" button, and give it a name.  This is then saved to your user account, so it will still be there if you log out, and log back in later, or on a different device.

To edit a search preset, click on the preset from the sidebar, make any changes you want, then click the "**Edit Preset**" button, then the "**Save Changes**" button.  To delete a search preset, click on it from the sidebar, then click the "**Delete Preset**" button.

### Search API

Programmatic searches are available via the [search_tickets](api.md#search_tickets) API (supports pagination and compact responses).




## Tips and Patterns

- **Release management**: Create a Release ticket and attach deploy, test, and rollback events. Upload your build artifacts to the ticket, then run deploy from the ticket. Artifacts automatically flow into the job.
- **Incident response**: Auto-create an Issue on alert fire with server context. Assign on-call, set a due date for follow-up, and track remediation steps with comments. Close when resolved.
- **Change control**: Use Change tickets for planned work. Attach validation jobs (pre-checks, post-checks) and require a second assignee to review.
- **Maintenance windows**: Schedule Maintenance tickets with due dates. Attach health-check jobs to verify post-maintenance status.
- **Runbooks**: Use the ticket body (Markdown) for runbooks and checklists. Link to jobs via ticket events for repeatable actions.


## Privileges

- [create_tickets](privileges.md#create_tickets): Create tickets.
- [edit_tickets](privileges.md#edit_tickets): Edit tickets, add comments, attach/remove files, run ticket events.
- [delete_tickets](privileges.md#delete_tickets): Permanently delete tickets.

Standard authentication applies for UI and API usage (sessions or API Keys). See [Privileges](privileges.md) and [API](api.md) for details.


## See Also

- Actions can create tickets automatically: see [Actions](actions.md)
- Ticket data model and field details: see [Ticket](data.md#ticket)
- Ticket API endpoints: see [API](api.md#tickets) ([search_tickets](api.md#search_tickets), [create_ticket](api.md#create_ticket), [update_ticket](api.md#update_ticket), [add_ticket_change](api.md#add_ticket_change), [update_ticket_change](api.md#update_ticket_change), [delete_ticket_file](api.md#delete_ticket_file), [delete_ticket](api.md#delete_ticket))
