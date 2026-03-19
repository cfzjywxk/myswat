# MCP Server Redesign

This document now describes the implemented direction of the `work` path:

- `myswat work` is a thin client that submits workflow requests to a local daemon
- `myswat server` runs that daemon and lazily starts managed workers per role
- workers register, claim assignments, submit artifacts, and publish verdicts through the daemon MCP endpoint

Legacy chat/session flows still exist elsewhere in the repo, but they are no longer the execution model for `work`.

## Summary

MySwat now evolves toward:

- `WorkflowKernel` as the deterministic stage orchestrator
- `myswat-server` as the MCP tool surface for knowledge, artifacts, review state, and structured handoffs
- backend-specific or hosted runtimes as execution adapters behind that tool surface

This keeps MySwat's current moat:

- stage control
- bounded review loops
- TiDB-backed shared memory
- traceable artifacts and verdicts

And removes the coupling where the workflow engine must manually shuttle prompts and outputs between agent subprocesses.

## Why Change

The old codebase was centered on CLI subprocess runners:

- [`SessionManager`](../src/myswat/agents/session_manager.py) owns lifecycle and CLI session restore
- [`AgentRunner`](../src/myswat/agents/base.py) is explicitly a subprocess abstraction
- [`factory.py`](../src/myswat/agents/factory.py) maps roles to `codex`, `claude`, and `kimi`
- [`settings.py`](../src/myswat/config/settings.py) hard-codes `*_path`, `*_default_flags`, and role-to-backend selection

That design worked, but it forced MySwat Python code to do two jobs:

1. Orchestration and workflow policy
2. Cross-agent communication plumbing

Those responsibilities should be separated.

## Design Goals

- Keep deterministic orchestration in Python.
- Move cross-agent coordination behind an MCP server boundary.
- Expose TiDB-backed memory, artifacts, and review state as MCP tools.
- Support CLI and future MCP-native runtimes through one runtime adapter layer.
- Preserve auditability and bounded loops.

## Non-Goals

- Do not turn MySwat into free-form agent swarm chat.
- Do not let agent-to-agent messaging replace workflow stage ownership.

## Compatibility Assumption

MySwat is still in quick development. The redesign should assume:

- no backward compatibility requirement
- TiDB backend can be reset
- old workflow/session rows do not need migration
- old CLI-coupled persistence shape does not need to survive

This materially changes the rollout strategy.

We should prefer a clean cutover over a long transitional architecture.

## Implemented Direction

```text
User / CLI
  |
  | local control API
  v
MySwat daemon
  |
  | starts workers, submits workflow jobs, tracks progress
  v
WorkflowKernel
  |
  | queue stage work, enforce limits, wait on persisted results
  v
myswat-server (MCP)
  |
  | tools: register_runtime, claim_next_assignment,
  |        complete_stage_task, publish_review_verdict, ...
  |
  +-- Managed worker: Codex
  +-- Managed worker: Claude
  +-- Managed worker: Kimi
  +-- Runtime: future MCP-native agent
```

## Responsibility Split

### MySwat daemon keeps ownership of

- local control API for CLI commands
- worker lifecycle and lazy startup
- workflow job submission

### WorkflowKernel keeps ownership of

- stage transitions
- stage entry prompts and constraints
- max iteration counts
- abort and escalation policy
- user checkpoints
- final success or failure decisions

### myswat-server owns

- structured agent coordination primitives
- shared knowledge and artifact access
- review-cycle creation and verdict persistence
- stage-local handoff state
- runtime discovery and capability advertisement

### Agent runtimes own

- local execution
- tool usage
- code editing and testing
- optional stage-local collaboration through server tools

## Communication Model

The server should expose structured workflow tools, not an unrestricted chat bus.

Implemented tool families:

### Knowledge

- `search_knowledge`
- `get_recent_artifacts`
- `get_work_item_snapshot`

### Workflow state

- `register_runtime`
- `heartbeat_runtime`
- `claim_next_assignment`
- `report_status`
- `submit_artifact`
- `complete_stage_task`
- `fail_stage_task`
- `persist_decision`

### Review loop

- `publish_review_verdict`
- internal orchestrator use of `request_review`

These primitives are intentionally narrower than generic `send_message`, and the public runtime-facing surface now ships through the stdio MCP server in [`mcp_stdio.py`](../src/myswat/server/mcp_stdio.py).

## Mapping to Existing MySwat State

The current TiDB schema is useful as a reference, but it should not constrain the redesign.

The existing schema already shows the right core entities:

- `work_items`
  - task ownership, current stage, process log
- `artifacts`
  - design docs, plans, patches, test plans, phase summaries
- `review_cycles`
  - proposer, reviewer, iteration, verdict
- `knowledge`
  - decisions, project ops, failure modes, conventions
- `sessions` and `session_turns`
  - agent-local conversation history

For the redesign, `myswat-server` should become the canonical persistence boundary.

The current [`MemoryStore`](../src/myswat/memory/store.py) is still a useful implementation reference and a temporary host for the new tool service, but after the cutover it should be reshaped around the new server-first schema rather than preserving old session semantics.

## Proposed Canonical Schema After Reset

After resetting TiDB, the canonical schema should be centered on workflow and coordination state rather than CLI session persistence.

Recommended tables:

- `projects`
  - project registry and repo metadata
- `agent_profiles`
  - architect, developer, qa roles and runtime preferences
- `work_items`
  - top-level task state and ownership
- `stage_runs`
  - one row per workflow stage execution with limits, owner, and status
- `coordination_events`
  - append-only structured records for status reports, handoffs, clarification requests, review requests, and escalations
- `artifacts`
  - submitted outputs under review
- `review_cycles`
  - review request and verdict state
- `knowledge`
  - distilled project knowledge and decisions
- `runtime_registrations`
  - available agent runtimes, capabilities, heartbeat, and lease state

Notably, `sessions` and `session_turns` should stop being the primary cross-agent state model.

If conversation persistence is still useful, it should be reintroduced later as a thin observability layer, not as the core coordination primitive.

## Execution Adapters

The current `AgentRunner` layer should become an execution adapter layer:

```text
WorkflowEngine
  -> StageController
    -> RuntimeAdapter
      -> CLI adapter or MCP runtime adapter
```

Recommended adapter contract:

- `start_stage(...)`
- `resume_stage(...)`
- `cancel(...)`
- `poll_status(...)`

Current CLI runners can be wrapped as one adapter implementation.

## Session Model

Today `SessionManager` mixes:

- TiDB session persistence
- context injection
- CLI session restore

The redesign should split that into:

- `SessionStore`
  - persists turns and stage metadata
- `RuntimeAdapter`
  - manages backend-specific session semantics
- `myswat-server`
  - exposes durable shared workflow state to any runtime

This change removes the assumption that cross-session continuity must come from CLI-specific resume behavior.

Because backward compatibility is not required, the better move is to delete `SessionManager` as a core concept from the new architecture and replace it with:

- `RuntimeAdapter`
- `StageRun`
- `CoordinationEvent`
- `Artifact`
- `ReviewCycle`

## Stage Flow Under the New Model

Example: `develop` stage

1. `WorkflowEngine` opens the stage with constraints, reviewer set, and exit conditions.
2. It starts the primary runtime for the stage leader.
3. The agent uses MCP tools from `myswat-server` to:
   - fetch context
   - report progress
   - submit artifacts
   - request review
   - respond to review feedback
4. Reviewer agents use the same server to fetch the artifact and publish verdicts.
5. `WorkflowEngine` watches the resulting persisted state and decides:
   - continue loop
   - escalate
   - abort
   - advance stage

The orchestrator no longer needs to manually relay dev output into QA prompts and back again.

## Schema Reset Implications

Since the backend can be reset, the redesign should also clean up accumulated complexity:

- remove migration compatibility logic that only exists to preserve old schema states
- replace CLI-session-specific columns with stage- and event-oriented records
- stop storing workflow coordination indirectly through prompt transcripts
- make `coordination_events` the auditable source of truth for handoffs and review loops

That is a better fit for MCP than trying to retrofit MCP semantics onto legacy session tables.

## Suggested MCP Tool Schemas

### `report_status`

Input:

- `work_item_id`
- `agent_id`
- `agent_role`
- `stage`
- `summary`
- `next_todos`
- `open_issues`

Effect:

- updates `work_items.metadata_json.task_state`
- appends a process log event

### `submit_artifact`

Input:

- `work_item_id`
- `agent_id`
- `iteration`
- `artifact_type`
- `title`
- `content`
- `stage`

Effect:

- creates or updates `artifacts`
- updates `last_artifact_id`
- appends a process log event

### `request_review`

Input:

- `work_item_id`
- `artifact_id`
- `iteration`
- `proposer_agent_id`
- `reviewer_agent_id`

Effect:

- creates a `review_cycles` row
- appends a process log event

### `publish_review_verdict`

Input:

- `cycle_id`
- `work_item_id`
- `reviewer_agent_id`
- `verdict`
- `issues`
- `summary`

Effect:

- updates `review_cycles.verdict`
- updates work item state
- appends a process log event

## Cutover Plan

### Step 1

- define the new server-first schema
- reset TiDB backend
- delete migration assumptions that only preserve old states

### Step 2

- implement `myswat-server` on top of the new schema
- expose structured MCP tools for workflow and knowledge access

### Step 3

- replace `SessionManager` with runtime adapters plus stage-run tracking
- make agent coordination flow through server tools only

### Step 4

- rewrite `WorkflowEngine` to observe `stage_runs`, `coordination_events`, `artifacts`, and `review_cycles`
- keep deterministic stage ownership and exit conditions

### Step 5

- delete obsolete CLI-session persistence code, old runner assumptions, and compatibility-only config

## Recommendation

MySwat should become a workflow kernel with an MCP tool surface.

It should not become a free-form multi-agent chat room.

The right boundary is:

- Python code decides who owns a stage, what counts as success, and when to stop
- `myswat-server` handles the shared workflow operations that agents use to collaborate
- runtime adapters hide whether an agent is reached via CLI, MCP, or a future API

And because backward compatibility is not required, the implementation should favor a clean schema reset and direct cutover rather than a long dual-stack transition.
