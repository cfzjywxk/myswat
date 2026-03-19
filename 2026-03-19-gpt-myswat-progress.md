# MySwat Progress Summary

Date: 2026-03-19
Authoring model: GPT

## Overview

MySwat was redesigned from a session/transcript-oriented workflow into an MCP-oriented workflow kernel with daemon-managed workers.

The main goal of the current work was:

- move orchestration and workflow state into MySwat itself
- make cross-role coordination MCP-shaped instead of subprocess-orchestrated chat
- make `myswat work` user-facing again through a daemon + managed worker model
- keep Codex and Claude CLIs as local worker execution adapters

## Architecture Progress

- Added a stage-oriented workflow kernel in `src/myswat/workflow/kernel.py`
- Added stateless workflow runtime role wrappers in `src/myswat/workflow/runtime.py`
- Added a store-backed MCP coordination layer in `src/myswat/server/`
- Added a managed worker loop in `src/myswat/cli/worker_cmd.py`
- Added a persistent local server command path through `myswat server`
- Refactored `myswat work` into a thin client that submits work to the daemon and follows progress

## Data Model Progress

The workflow model was redesigned around explicit orchestration state rather than conversational session state.

Key workflow entities now include:

- `stage_runs`
- `coordination_events`
- `runtime_registrations`
- persisted review cycles and artifacts tied to workflow stages

This supports deterministic stage ownership, persisted handoff state, worker claiming, and review-loop auditability.

## Worker / MCP Progress

Managed workers now:

- register with the MySwat server
- heartbeat periodically
- claim assignments over MCP
- execute the assigned role locally with Codex or Claude
- publish stage completions and review verdicts back through MCP

The worker path is now file-aware for large payloads:

- large prompts are externalized to `/tmp/*.md`
- large system context is externalized to `/tmp/*.md`
- workers instruct the agent CLI that file-backed payloads are allowed
- large responses can come back as file references
- file-referenced stage outputs and review verdict payloads are resolved back into canonical text before persistence

This now gives a full file-backed roundtrip at the worker boundary.

## Workflow / UX Progress

The intended user flow is now represented in code:

1. `myswat server`
2. `myswat init <project> --repo <path>`
3. `myswat work -p <project> "<requirement>"`
4. `myswat status -p <project> --details`

Foreground `myswat work` behavior was also improved so it can keep following workflow progress and cancel the work item on `Ctrl-C`.

## Testing Progress

Coverage was added for:

- MCP stdio and HTTP tool transport
- managed worker assignment roundtrips
- daemon lifecycle and worker supervision behavior
- prompt construction across design / review / implementation stages
- large-payload prompt and context externalization
- large-payload response resolution
- end-to-end fake-runtime fib-demo workflow execution
- optional live Codex smoke path

Important regression coverage now exists for:

- design / design review prompt construction
- implementation / code review prompt construction
- worker prompt/context externalization
- worker stage response file resolution
- worker review verdict JSON file resolution
- kernel JSON payload resolution for file-backed responses

## Current Practical State

What is working:

- MySwat has an MCP-oriented control plane
- the daemon owns orchestration
- workers are daemon-managed
- workers use local Codex / Claude CLIs as execution adapters
- the fib-demo workflow has automated E2E coverage in tests
- large payload transport no longer depends on aggressive prompt truncation

Known boundary:

- live provider-backed E2E was not executed inside the sandbox during this work
- automated E2E currently proves the workflow with fake workers plus targeted live-smoke coverage rather than a fully provider-backed multi-role CI test

## Verification Snapshot

Focused verification completed during this work included combinations of:

- `pytest` for worker, server, kernel, and fib-demo integration suites
- `compileall` on touched modules

Recent targeted result:

- worker/server/kernel/fib-demo focused suite: passed
- file-backed roundtrip regression tests: passed

