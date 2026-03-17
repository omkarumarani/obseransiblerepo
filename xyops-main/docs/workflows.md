# Workflows

## Overview

Workflows in xyOps are visual graphs of nodes connected by wires that control job execution. Conceptually, a workflow is an event with an embedded graph and some special runtime behavior. When a workflow runs, it becomes a "workflow job" which may spawn any number of sub-jobs on connected nodes. The system manages their lifecycle, collects results, applies actions and limits, and tracks everything in the parent job record.

How a workflow runs, at a glance:

- A trigger node fires (e.g. manual trigger, schedule trigger, etc.). That starts the workflow job and "lights" the trigger node.
- All nodes wired from that trigger activate. Multiple outputs run in parallel unless constrained by limits.
- Event or job nodes launch sub-jobs. Their completion result determines which outgoing wires fire next.
- Controller nodes implement fan-out/fan-in, loops, conditionals, delays and multiplexing.
- Action and limit nodes visually attach to event/job nodes and are merged into the launched sub-jobs.
- The workflow completes when no nodes are active and all sub-jobs are finished or aborted. The parent job's result summarizes overall success/warnings/failures.

Workflows are powerful because they combine reusable events, ad-hoc jobs, resource limits, job actions, and flow control on one graph. They run everything in parallel by default, with concurrency governed by resource limits.


## When To Use Workflows

- **Orchestration**: Coordinate multiple jobs with conditional logic, joins, and/or delays.
- **Fan-out processing**: Split a dataset or file list and process items concurrently, then join results.
- **Multi-target runs**: Run the same job on many servers (multiplex) with optional staggering.
- **Reusability**: Compose pre-defined events into larger flows, or create ad-hoc job nodes when one-offs are easier.
- **Post-processing**: Attach actions directly to nodes with conditions, including success, failure, warnings, and custom tags.


## Graph Editor

The workflow editor provides:

- **Connect Nodes**: Click pole buttons to solder connections. Invalid pairs are suppressed by the UI. Conditions appear inline on wires; click to change.
- **Add Node**: Click "Add Node" or solder from a pole and click the background to insert a new node in place.
- **Duplicate**: Select one or more nodes (shift-click) and duplicate; connections between selected nodes are preserved.
- **Detach**: Detach all connections to/from selected nodes.
- **Delete**: Delete selected nodes and any connected wires. Deleting a trigger node also removes the underlying trigger.
- **Undo/Redo**: Up to 100 levels for all editor operations.
- **Zoom/Scroll**: Zoom in/out/reset and drag to pan.
- **Test Selection**: Run a test starting from the selected node or only that single node. Optional: disable actions/limits, provide custom input JSON and/or upload files.
- **Standard Buttons**: Cancel, Export, History, Save Changes.


## Node Types

Nodes have connection "poles" on their sides: an input pole on the left (incoming flow), an output pole on the right (outgoing flow), and for event/job nodes a special limit pole on the bottom used to attach limit nodes. Poles can connect to multiple nodes unless the controller type restricts its output as noted below.

### Trigger Nodes

Trigger nodes visually represent event triggers inside the graph, such as manual, schedule, interval, webhook or plugin triggers (see [Triggers](triggers.md)). They have a single output and typically feed Event, Job, or Controller nodes.

Special trigger option bubbles (Catch-Up, Range, Blackout, Delay, Precision) have no poles and are purely modifiers that light up when their associated scheduled trigger fires; they do not connect to anything.

### Event Nodes

Event nodes place a pre-created event on the graph. You can override targets, algo, tags and user parameters for that use. When the node runs, the sub-job inherits the event's configuration plus the node's overrides. If the referenced event is a workflow, it must include an enabled manual trigger so the engine knows where to start the sub-workflow. 

Event nodes present input, output and limit poles and can accept flow from triggers, other event/job nodes or controllers, and can send flow to other event/job nodes, actions, or controllers.

### Job Nodes

Job nodes are ad-hoc jobs without a backing event. You choose a plugin and provide any parameters, plus optional title/icon/category, targets, algo and tags. Actions and limits attached in the graph are merged into the launched sub-job at runtime. Job nodes also include input, output and limit poles, accept flow from triggers/event/job/controllers, and can send to event/job/action/controller nodes.

See [Event vs Job Nodes](#event-vs-job-nodes) below.

### Action Nodes

Action nodes attach post-job actions to event/job nodes and are merged into the launched sub-job with the selected condition. Typical uses include email notifications, webhooks, or disabling/deleting future runs (see [Actions](actions.md)).

Action nodes have a single input and connect from event/job nodes.

### Limit Nodes

Limit nodes attach resource controls to event/job nodes and are merged into the launched sub-job's limits (see [Limits](limits.md)). Examples include Max Concurrent Jobs, Max Queue Size, CPU/Memory/Time and File inputs. 

A limit node connects via the bottom limit pole on an event/job node.

### Controller Nodes

Controllers implement flow control. They generally have input and output poles and connect in-line between other nodes. Some controllers require a single output connection; details are in the controller sections below.

#### Split Controller

The Split controller fans out work by taking an input list and launching one sub-job per item. Provide a dot-path to the list in the previous job context, such as `data.rows`. The engine resolves the path and expects an array; if the value is a string it is trimmed and split by newline.   A special case is `files`, which splits the incoming files array so that each sub-job receives exactly one file. 

In the UI, the split controller configuration dialog provides an "Expression Builder" button, which allows you to explore output data from recently completed jobs, and pick out a specific JSON key path to use for the expression string.

Each individual sub-job will receive one item from the split data.  It will arrive in the job's [Job.input](data.md#job-input), either in `data` as a property named `item`, or as a [File](data.md#file) in the `files` array.

Split requires exactly one output connection to the Event or Job node it will run per item. Concurrency and queuing are governed by the limits attached to that node. After all items complete, you can continue the flow using a `continue` wire from the controlled node. The controller includes a "continue percentage" setting so you can require that at least N% of the sub-jobs succeed before continuing.

#### Join Controller

The Join controller waits for multiple incoming flows to finish, then passes a combined result to the next step. You can wire multiple inputs into a Join; it initializes when the first input arrives and completes after all of its inputs have fired.

The joining process works as follows: all input job data is appended to an `items` array, and also separately all job data is shallow-merged into a `combined` object, which is passed to the next job (via continue condition).  For e.g. if 3 connected input jobs all output this data: `{"foo":1234}` then the final joined data that is passed along would look like this:

```json
{
	"items": [
		{ "foo": 1234 },
		{ "foo": 1234 },
		{ "foo": 1234 }
	],
	"combined": {
		"foo": 1234
	}
}
```

Any files produced upstream are concatenated and passed along. Join requires exactly one output connection.

#### Repeat Controller

The Repeat controller runs the same Event or Job node a fixed number of times. You configure the iteration count on the controller. All runs are launched immediately and will queue or run in parallel based on the limits attached to the target node. Repeat requires exactly one output connection (to the node being repeated). After all iterations complete, use a `continue` wire from the repeated node to define post-processing.

To control the series/parallel run of the repeat jobs, the user simply has to connect limit nodes to the event/job node, e.g. "Max Concurrent Jobs" and "Max Queue Size".

The repeat controller also offers a "continue percentage" text field, where the user can enter a number from 0-100.  This represents the number of sub-jobs that must complete successfully for the controller to fire the "continue" condition and allow flow to continue (otherwise the workflow will end, assuming no other nodes are active).

#### Multiplex Controller

The Multiplex controller runs a job across many servers. It expands the target selection from the destination Event/Job into concrete server IDs, filters to currently enabled servers, and applies server alert filters. It then launches one sub-job per server. An optional stagger setting delays starts by a fixed interval per job, which helps avoid thundering herds. Multiplex requires exactly one output connection (to the Event/Job to be run per server).

To control the series/parallel run of the multiplexed jobs, the user simply has to connect limit nodes to the event/job node, e.g. "Max Concurrent Jobs" and "Max Queue Size".

The multiplex controller also offers a "continue percentage" text field, where the user can enter a number from 0-100.  This represents the number of sub-jobs that must complete successfully for the controller to fire the "continue" condition and allow flow to continue (otherwise the workflow will end, assuming no other nodes are active).

#### Decision Controller

The Decision controller evaluates a JEXL expression (with [xyOps extensions](xyexp.md)) against the previous job context, for example `data.random > 0.5`. If the expression evaluates to true, the controller passes control to all of its connected outputs; when false, no outputs fire. There is no explicit "false" pole. For multi-branch logic, create multiple Decision nodes with different expressions and optional titles/icons to make branches visually clear.

In the UI, the decision controller configuration dialog provides an "Expression Builder" button, which allows you to explore output data from recently completed jobs, and pick out a specific JSON key path to use for the expression string.

Decision does not require a single output and can fan out to many.

#### Wait Controller

The Wait controller pauses flow for a configured duration, then passes control to all connected outputs. It maintains active state while waiting and is aborted if the workflow is aborted.

Wait does not require a single output and can feed multiple downstream steps.

### Note Nodes

A Note node is simply a customizable text box you can use to annotate your workflows.  Enter your note's body content using [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/), and it will display on your workflow map.  You can drag notes around to position them wherever you like, and even make a double-wide note.

By default, notes are not shown during job runs -- they are more for providing instructions to users who are configuring the workflow.  However, a checkbox is provided to always show the note, even during job runs.

## Connections and Conditions

Here are the connection rules by node type: 

- Triggers send flow to Event, Job, or Controller nodes.
- Event and Job nodes accept flow from Triggers, Event/Job, or Controller nodes and can send to Event/Job, Action, or Controller nodes; their bottom limit pole accepts Limit nodes. 
- Action nodes connect from Event/Job nodes only. 
- Limit nodes attach to the bottom pole of Event/Job nodes. 
- Controllers accept flow from Trigger/Event/Job nodes and send to Event/Job nodes.

Controller output restrictions:

- Split, Join, Repeat, Multiplex: must have exactly one output connection (to the node being controlled or the post-join node).
- Decision and Wait: may have multiple outputs.

Conditions on wires from Event/Job nodes determine which outputs fire when a sub-job completes. Supported values include complete (always), success (code 0), error (any failure), the specific codes warning, critical, or abort, tag:NAME (fires if the sub-job produced tag NAME), and continue (a special condition fired after Repeat/Multiplex/Split completes when the success threshold is met).

The editor defaults new wires from Event/Job outputs to success, and you can change the condition inline on the wire (just click it). Note that Action and Limit nodes do not forward flow: they are attached to the launched sub-job; actions require a condition and limits attach via the bottom pole.


## Continue After Controllers

Repeat, Multiplex and Split support a "continue after controller" flow:

- Configure the controller's "continue percentage" (0-100). This is the minimum percentage of successful sub-jobs required to proceed.
- Solder wires out of the controlled Event/Job node with the `continue` condition to define what happens next.
- When all iterations complete, if successes meet or exceed the threshold, all `continue` wires from that Event/Job fire. Otherwise nothing continues and the workflow may finish if no other nodes are active.

Example: Event A → Repeat (x10) → Event A has a `continue` wire → Event B. After all 10 runs of Event A finish, if at least N% succeed (per controller), Event B runs.


## Data Passing Between Nodes

Inputs and outputs are automatically passed along:

- At workflow start, the inbound `input.data` and `input.files` are passed to the trigger node.  The trigger nodes passes these on to any soldered nodes.
- When an Event/Job finishes, its output `data` and `files` are passed to downstream nodes as `input.data` and `input.files`.
- If the workflow itself has any user fields defined, these are passed to all sub-jobs via a `workflow.params` object inside the job data.
- Tags: user tags from sub-jobs bubble up to the workflow job and can drive `tag:...` conditions.
- HTML and table content: if a sub-job emits `html` or `table`, it bubbles up to the parent for display.  If multiple jobs emit content The latter prevails.
- Retries: if a sub-job was retried, its data/files are not bubbled and it doesn't count toward tag/condition firing.

Join specifics:

- The next node after a Join Controller receives a custom `input.data` with two properties: `items` (array of each upstream job's data) and `combined` (shallow merge of all data).
- Any files are concatenated onto the `input.files` array.

Split specifics:

- The Split controller resolves its data path against the previous job context (or incoming input).
- If the path is `files`, each incoming file is sent to a separate sub-job; otherwise, each array element becomes `input.data = { item: ... }` for the launched sub-job.


## Event vs Job Nodes

When should you use an Event vs a Job Node:

- **Event node**: References a pre-created event. Use this when you want reusable configuration: plugin params, targets, algo, actions, limits, and optional user fields. The node can override targets, algo, tags, and user params per use.
- **Job node**: Ad-hoc job without an event. You pick the plugin and fill plugin parameters right in the workflow editor (e.g. write inline shell script with the Shell plugin). This is faster for one-offs or when you don't want a separate event.


## Sub-Workflows

Event nodes can reference events of type `workflow`. To run as a sub-workflow:

- The sub-workflow must have an enabled manual trigger. The engine uses this to determine the start node.
- Parent workflow input (data/files) is passed into the sub-workflow's trigger node.
- Limits and actions on the parent workflow do not automatically apply to the sub-workflow unless explicitly attached to the node.


## Tips

- **Reuse vs. ad-hoc**: Use Event nodes to encapsulate stable configuration and user fields, and Job nodes for quick one-offs or inline scripts.
- **Concurrency control**: Everything runs in parallel by default; attach [Max Concurrent Jobs](limits.md#max-concurrent-jobs) and [Max Queue Limit](limits.md#max-queue-limit) limits to throttle fan-out.
- **Post-controller flow**: For Repeat/Multiplex/Split, use `continue` wires from the controlled node to handle "after all done" steps, optionally with a success threshold.
- **Fan-in**: Use Join to aggregate multiple upstream results; the next node sees both `items` and a `combined` object.
- **Condition routing**: Prefer `success`/`error` wires for the main paths and add `warning`, `critical`, `abort`, or `tag:NAME` for special handling.
- **Conditional logic**: Use Decision for branching; duplicate the node for multi-branch flows and give each a clear title/icon.
- **Staggering**: For massive multi-server jobs, add a Multiplex controller with a stagger to avoid spikes.
- **File workloads**: Split on `files` to process one file per sub-job; combine results with Join.


## Data Model Reference

Workflows are stored inside events. See [Data Structures](data.md) for complete schemas. Highlights:

- [Workflow](data.md#workflow): `{ start?, nodes: [], connections: [] }`.
- [Nodes](data.md#workflownode): `{ id, type: 'trigger'|'event'|'job'|'limit'|'action'|'controller', x, y, data? }`.
- [Connections](data.md#workflowconnection): `{ id, source, dest, condition? }` where `condition` matches the wire condition list above.
- Controller data:
  - Multiplex: `{ controller: 'multiplex', stagger?, continue? }`
  - Repeat: `{ controller: 'repeat', repeat, continue? }`
  - Split: `{ controller: 'split', split: 'data.path.or.files', continue? }`
  - Join: `{ controller: 'join' }`
  - Decision: `{ controller: 'decision', decision: '<expression>' , title?, icon? }`
  - Wait: `{ controller: 'wait', wait }`


## Security and Privileges

Creating or editing a workflow is subject to the same privileges as events. When a workflow runs sub-jobs, target and category privileges still apply to those launched jobs. 

See [Privileges](privileges.md) and [Events](events.md) for more details.


## API

Workflows reuse the event APIs -- there are no separate workflow APIs. In particular:

- [get_events](api.md#get_events)
- [get_event](api.md#get_event)
- [create_event](api.md#create_event)
- [update_event](api.md#update_event)
- [run_event](api.md#run_event)
- [delete_event](api.md#delete_event)

See [API → Events](api.md#events) for details.

## Notes and Caveats

- Controller single-output requirement: Split, Join, Repeat, and Multiplex controllers must have exactly one output.
- Sub-workflow manual trigger: A sub-workflow must have an enabled manual trigger or it cannot be launched from a workflow node.
- Action/Limit nodes: These do not forward flow; they are merged into the launched sub-job and executed/checked there.
- Modifier triggers: Catch-Up, Range, Blackout, Delay, Precision are visual modifiers only and are "lit" when their associated schedule fires; they do not connect to other nodes.


## See Also

- [Events](events.md)
- [Actions](actions.md)
- [Limits](limits.md)
- [Triggers](triggers.md)
- [Plugins → Event Plugins](plugins.md)
- [xyOps Expression Format](xyexp.md)
