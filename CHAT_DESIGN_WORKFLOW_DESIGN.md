# Chat-Triggered Design and Test-Plan Workflows

## Review Feedback Incorporated

### Comment addressed

- `[qa_main] review failed (exit=1)`

### Resolution

No substantive design criticism was returned from QA; the reviewer process exited
before producing structured feedback. I therefore treat this as a **workflow
execution-failure gap** in the design rather than a disagreement with the core
architecture.

This updated design now explicitly defines:

- how proposer/reviewer agent execution failures are classified
- when a workflow should be marked `blocked` versus `review`
- how process-log events and task state should record agent failures
- how reports should surface failure reasons
- targeted tests for reviewer/proposer non-zero exits

## Summary

This design adds two chat-native teamwork flows without introducing new slash
commands or a second orchestration stack:

- **Architect-led design review**: architect proposes a formal design, then
  developer plus available QA reviewers iterate until all reviewers return LGTM.
- **QA-led test-plan review**: the active QA agent proposes a formal test plan,
  then architect plus developer iterate until both reviewers return LGTM.

The feature is implemented by extending the existing `````delegate````` block
with a `MODE:` field, forking the proposer into a work-item-scoped TiDB session,
and reusing `WorkflowEngine` for the actual review lifecycle, persistence,
process logs, artifacts, and user checkpoints.

This document is the technical design proposal for implementing chat-triggered
team design and test-plan workflows in MySwat.

## Goals

- Let the architect trigger a team design-finalization workflow directly from
  `myswat chat`.
- Let either QA agent trigger a team test-plan-finalization workflow directly
  from `myswat chat`.
- Preserve the existing natural-language delegation pattern; no new slash
  commands are added.
- Ensure work-item, artifact, review-cycle, and process-log state is fully
  persisted and visible in `myswat status` / `myswat task --details`.
- Preserve the original chat session so `/chat` resume semantics do not break.
- Reuse existing workflow infrastructure wherever possible.
- Distinguish **tooling/runtime failures** from **substantive review feedback**.

## Non-Goals

- No new public `myswat work` mode flags are added for these chat-only flows.
- No changes are made to the existing full workflow stage order.
- No new review-cycle schema is introduced; existing artifact and review-cycle
  persistence is reused.
- No attempt is made in v1 to clone an AI CLI session into a second independent
  runner; the proposer workflow reuses the current runner in-process.

## Current State

### Chat delegation today

`src/myswat/cli/chat_cmd.py` currently supports a single delegation path:

- `_extract_delegation()` returns only a task string.
- Architect responses containing a `````delegate````` block automatically start
  `_run_inline_review_interactive()`.
- The inline review loop always routes to developer + `qa_main` and creates a
  `code_change` work item.

### Workflow engine today

`src/myswat/workflow/engine.py` already supports multi-stage workflows through
`WorkMode.full`, `WorkMode.design`, `WorkMode.development`, and
`WorkMode.test`.

Important existing capabilities we should reuse:

- persistent `work_items.metadata_json.task_state`
- process-log events via `append_work_item_process_event()`
- artifact persistence via `create_artifact()`
- review persistence via `create_review_cycle()` + `update_review_verdict()`
- multi-reviewer `_run_review_loop()` with overridable proposer/reviewers
- proposer-aware `_user_checkpoint(..., proposer=...)`

### Session model today

`SessionManager` persists TiDB session turns while the underlying AI CLI keeps
its own resumable session ID. `create_or_resume()` binds sessions by
`(agent_id, work_item_id)` and `close()` only closes/compacts the TiDB session;
it does not reset the runner. That makes runner reuse safe for a forked
work-item session.

## Requirements and User Flows

### Flow A: architect-led design finalization

1. User chats with `architect`.
2. Architect decides the discussion is ready for team review and emits:

   ````text
   ```delegate
   MODE: design
   TASK: finalize the caching architecture based on our discussion
   ```
   ````

3. Chat detects `MODE: design` and starts an architect-led workflow.
4. A new `design` work item is created.
5. The architect chat session is **forked** into a work-item-scoped TiDB
   session that reuses the same AI runner.
6. Architect proposes the formal design.
7. Developer plus all available QA agents review until all return LGTM or the
   iteration limit is reached.
8. User sees the reviewed design and can approve, reject, or send feedback.
9. Workflow report is persisted; control returns to the original architect chat
   REPL.

### Flow B: QA-led test-plan finalization

1. User chats with `qa_main` or `qa_vice`.
2. QA decides the discussion is ready for team review and emits:

   ````text
   ```delegate
   MODE: testplan
   TASK: finalize the release test plan for the authentication work
   ```
   ````

3. Chat detects `MODE: testplan` and starts a QA-led workflow.
4. A new work item is created and marked `in_progress`.
5. The active QA chat session is forked into a work-item-scoped TiDB session
   that reuses the same AI runner.
6. QA proposes the formal test plan.
7. Architect plus developer review until both return LGTM or the iteration
   limit is reached.
8. User sees the reviewed plan and can approve, reject, or send feedback.
9. Workflow report is persisted; control returns to the original QA chat REPL.

## Architecture Overview

### High-level approach

We keep one orchestration substrate and add two **chat-only internal workflow
modes**:

- `architect_design`
- `testplan_design`

The architecture is:

1. **Agent prompt emits typed delegation**
2. **`chat_cmd` parses delegation mode + task**
3. **Chat dispatcher validates role/mode combination**
4. **Interactive wrapper creates work item and session fork**
5. **`WorkflowEngine` runs a single-artifact review workflow**
6. **Existing persistence paths record artifacts, review cycles, task state,
   and process logs**
7. **Wrapper closes forked workflow sessions and returns to the original REPL**

### Why reuse `WorkflowEngine`

The engine already owns the hardest parts of the workflow contract:

- consistent state transitions
- review-loop persistence
- cancellation handling
- report generation
- user checkpoints
- process-log history used by `myswat status --details`

Reusing it prevents a second implementation of review persistence inside
`chat_cmd` and keeps future review behavior changes centralized.

## Detailed Design

### 1. Delegation format and parsing

#### Syntax

Supported delegate block:

````text
```delegate
MODE: <code|design|testplan>
TASK: <task text>
```
````

`MODE:` is optional.

- Missing `MODE:` defaults to `code`.
- `TASK:` remains the canonical task field.
- If `TASK:` is absent, fallback remains the block body, matching current
  behavior.

#### Parsing contract

`src/myswat/cli/chat_cmd.py`

```python
DelegationMode = Literal["code", "design", "testplan"]

def _extract_delegation(text: str) -> tuple[str, DelegationMode] | None:
    """Return (task, mode) or None."""
```

#### Validation rules

- Accepted modes: `code`, `design`, `testplan`
- Unknown mode: do **not** auto-run a workflow; print a warning and leave the
  chat session active
- Role/mode matrix:
  - `architect` + `code` => existing dev+QA inline review
  - `architect` + `design` => architect-led design workflow
  - `qa_main|qa_vice` + `testplan` => QA-led test-plan workflow
  - all other combinations => warn and ignore the delegate block

This avoids accidentally routing malformed or unsupported delegation blocks into
an incorrect workflow.

### 2. Session strategy: fork, do not rebind

#### Problem

The active proposer chat session has `work_item_id = NULL`. If the workflow runs
inside that same session, the work item has no proposer-linked session and
`myswat task --details` cannot show the proposer’s workflow turns correctly.

#### Decision

Add `SessionManager.fork_for_work_item()`.

```python
def fork_for_work_item(
    self,
    work_item_id: int,
    purpose: str | None = None,
) -> "SessionManager":
    ...
```

#### Behavior

- create a **new TiDB session** with `work_item_id=<new item>`
- set `parent_session_id` to the original chat session ID when available
- reuse the **same `AgentRunner` instance**
- leave the original chat session open and untouched

#### Why this is correct

- work-item details now include proposer workflow turns
- original REPL session remains resumable
- proposer still retains in-process discussion context because the runner is
  shared
- no historical rows are mutated after the fact

#### Lifecycle

- original chat session: remains active throughout
- forked workflow session: closed at workflow end
- shared runner: remains alive and attached to the original chat manager after
  the workflow returns

#### Trade-off

Because CLI session IDs are currently persisted through assistant-turn metadata,
not a dedicated session column, a user who exits immediately after the workflow
**before another proposer turn** may resume later from the pre-workflow CLI
session marker. This is acceptable for v1 because workflow turns are persisted
in the forked session and the in-process chat continuity works correctly.
If this becomes user-visible, a follow-up should add a session-level persisted
CLI session ID.

### 3. New chat wrappers

`src/myswat/cli/chat_cmd.py`

Add two interactive wrappers analogous to `_run_workflow_interactive()`:

- `_run_design_review_interactive(...)`
- `_run_testplan_review_interactive(...)`

Each wrapper should:

- reuse `_run_with_task_monitor()` so ESC/cancel behavior matches existing chat
  workflows
- create the work item before the worker thread begins
- register cancel targets for every runner created inside the workflow
- update final work-item status to `completed`, `review`, or `blocked`

#### 3.1 Architect-led design wrapper

Suggested internal signature:

```python
def _run_design_review_interactive(
    store: MemoryStore,
    proj: dict,
    proposer_sm: SessionManager,
    compactor: KnowledgeCompactor,
    workdir: str | None,
    settings: MySwatSettings,
    task: str,
    prompt_session: PromptSession | None = None,
) -> None:
    ...
```

Behavior:

- validate required agents: `architect` proposer already exists, `developer`
  exists, at least one QA exists
- create work item with:
  - `item_type="design"`
  - `assigned_agent_id=architect.id`
  - `metadata_json={"work_mode": "architect_design"}`
- mark `in_progress`
- fork proposer: `arch_workflow_sm = proposer_sm.fork_for_work_item(...)`
- create/resume reviewer sessions for developer and all available QA roles with
  `work_item_id=<id>`
- create mode-aware `ask_user` callback reusing the same prompt-toolkit pattern
  used in `_run_workflow()`
- instantiate `WorkflowEngine(..., mode=WorkMode.architect_design, arch_sm=arch_workflow_sm, auto_approve=False)`
- run `engine.run(task)`
- close only workflow-scoped sessions; do **not** close the original proposer
  chat session

#### 3.2 QA-led test-plan wrapper

Suggested internal signature:

```python
def _run_testplan_review_interactive(
    store: MemoryStore,
    proj: dict,
    proposer_sm: SessionManager,
    compactor: KnowledgeCompactor,
    workdir: str | None,
    settings: MySwatSettings,
    task: str,
    prompt_session: PromptSession | None = None,
) -> None:
    ...
```

Behavior:

- validate required agents: proposer QA exists, `architect` exists,
  `developer` exists
- create work item with:
  - `item_type="review"`
  - `assigned_agent_id=<qa proposer>`
  - `metadata_json={"work_mode": "testplan_design"}`
- mark `in_progress`
- fork proposer QA session into work-item scope
- create architect + developer review sessions with `work_item_id=<id>`
- instantiate `WorkflowEngine(..., mode=WorkMode.testplan_design, arch_sm=arch_sm, qa_sms=[qa_workflow_sm], auto_approve=False)`
- run and close only workflow-scoped sessions

### 4. Workflow engine changes

`src/myswat/workflow/engine.py`

#### 4.1 New internal modes

Extend `WorkMode`:

```python
class WorkMode(StrEnum):
    full = "full"
    design = "design"
    development = "development"
    test = "test"
    architect_design = "architect_design"
    testplan_design = "testplan_design"
```

These modes are internal implementation modes used by chat wrappers; they are
not exposed as new top-level CLI flags.

#### 4.2 Engine constructor

Add an optional architect session manager:

```python
def __init__(
    self,
    store: MemoryStore,
    dev_sm: SessionManager,
    qa_sms: list[SessionManager],
    project_id: int,
    work_item_id: int | None = None,
    max_review_iterations: int = 5,
    mode: WorkMode = WorkMode.full,
    ask_user: Callable[[str], str] | None = None,
    auto_approve: bool = False,
    should_cancel: Callable[[], bool] | None = None,
    arch_sm: SessionManager | None = None,
) -> None:
    ...
```

`arch_sm` is required only for `architect_design` and `testplan_design`.
Construction should fail fast with `ValueError` if those modes are selected
without an architect session.

#### 4.3 Mode-aware startup state

`run()` currently records `workflow_started` as if developer owns every flow.
That is incorrect for the two new modes.

Required behavior:

- `full|design|development|test`: keep current developer-owned startup event
- `architect_design`:
  - `next_todos=["Architect produce design"]`
  - initial `to_role="architect"`
  - `updated_by_agent_id=self._arch.agent_id`
- `testplan_design`:
  - `next_todos=["QA produce test plan"]`
  - initial `to_role=<qa proposer>`
  - `updated_by_agent_id=<qa proposer id>`

This ensures `myswat status --details` reflects the actual proposer.

#### 4.4 New mode execution paths

##### `_run_architect_design_mode(requirement, result)`

Stages:

1. Architect produces design using new `ARCH_TECH_DESIGN`
2. Persist `design_draft` and process event from architect to reviewers
3. Run `_run_review_loop()` with:
   - `artifact_type="arch_design"`
   - `proposer=self._arch`
   - `reviewers=[self._dev, *self._qas]`
   - `abort_on_agent_failure=True`
4. If review fails:
   - `result.design_review_passed = False`
   - `result.success = False`
   - persist stage `design_review_failed` or `design_review_blocked`
   - generate final report and return
5. Show user checkpoint using `_user_checkpoint(..., proposer=self._arch)`
6. If user rejects, persist stage `design_rejected_by_user`, set
   `result.success = False`, generate final report, return
7. If user gives feedback, architect revises in-place via the checkpoint helper
8. Persist final approved design, set `result.success = True`
9. Generate a concise design-workflow report

##### `_run_testplan_design_mode(requirement, result)`

Stages:

1. QA proposer generates test plan using new `QA_DESIGN_TEST_PLAN`
2. Persist `testplan_draft` and process event from QA to architect/developer
3. Run `_run_review_loop()` with:
   - `artifact_type="test_plan"`
   - `proposer=<qa proposer>`
   - `reviewers=[self._arch, self._dev]`
   - `abort_on_agent_failure=True`
4. If review fails, persist `testplan_review_failed` or
   `testplan_review_blocked`, set `result.ga_test.test_plan_review_passed = False`,
   and return with report
5. Show user checkpoint using `_user_checkpoint(..., proposer=<qa proposer>)`
6. If user rejects, persist `testplan_rejected_by_user`, set
   `result.success = False`, generate final report, return
7. If user supplies feedback, QA revises the test plan in-place via checkpoint
8. Persist final approved plan and set `result.success = True`
9. Generate concise report through a dedicated branch in `_generate_report()`

#### 4.5 Agent execution-failure semantics

This section is the main update driven by the failed QA review attempt.

##### Problem

A review failure like `[qa_main] review failed (exit=1)` is not actionable
feedback for the proposer. The system should not ask the proposer to revise a
design or plan in response to a reviewer tool crash, network problem, or CLI
failure.

##### Decision

Extend `_run_review_loop()` with an optional strict mode:

```python
def _run_review_loop(
    self,
    artifact: str,
    artifact_type: str,
    context: str = "",
    proposer: SessionManager | None = None,
    reviewers: list[SessionManager] | None = None,
    abort_on_agent_failure: bool = False,
) -> tuple[str, int, bool]:
    ...
```

##### Rules

When `abort_on_agent_failure=True`:

- if any reviewer returns `response.success == False`:
  - append a `review_failure` process event with role and exit code
  - persist task state with `current_stage=f"{artifact_type}_review_blocked"`
  - set `open_issues=[f"[{role}] review failed (exit={code})"]`
  - stop the review loop immediately
  - **do not** ask the proposer to address comments
- if the proposer fails while drafting or addressing review feedback:
  - append a `proposal_failure` or `revision_failure` process event
  - persist a `*_blocked` stage
  - return a blocked workflow result

When `abort_on_agent_failure=False`, existing behavior remains unchanged for
already-implemented flows.

##### Result model

To let wrappers set a correct final work-item status, extend `WorkflowResult`
with lightweight failure metadata:

```python
@dataclass
class WorkflowResult:
    ...
    blocked: bool = False
    failure_summary: str = ""
```

Semantics:

- `blocked=True` means workflow execution was interrupted by a tooling/runtime
  failure or cancellation-like failure, not by substantive review rejection.
- `failure_summary` stores a short user-visible explanation for reports and task
  details.

##### Wrapper behavior

Interactive wrappers should map final status as:

- `blocked` if `cancelled` or `result.blocked`
- `completed` if `result.success`
- `review` otherwise

This ensures a reviewer process crash does not appear as normal review churn.

### 5. Review-loop prompt selection

#### Problem

`_build_review_prompt()` currently hardcodes reviewer identity by artifact type:

- `design` => QA prompt
- `test_plan` => developer prompt

That is incorrect for the new chat flows where:

- developer reviews an architect-authored design
- architect reviews a QA-authored test plan

#### Decision

Keep the current artifact flow and extend review-prompt selection to be
**reviewer-aware**.

```python
def _build_review_prompt(
    self,
    artifact_type: str,
    context: str,
    artifact: str,
    iteration: int,
    reviewer: SessionManager | None = None,
) -> str:
    ...
```

Call site inside `_run_review_loop()` becomes:

```python
prompt = self._build_review_prompt(
    artifact_type,
    context,
    current,
    iteration,
    reviewer=reviewer,
)
```

#### Prompt rules

- existing developer-led design workflow keeps `QA_DESIGN_REVIEW`
- existing developer-led test-plan review keeps `DEV_REVIEW_TEST_PLAN`
- new architect-led design workflow uses new role-neutral `DESIGN_REVIEW`
- new QA-led test-plan workflow uses new role-neutral `TEST_PLAN_REVIEW`

This preserves backward compatibility while enabling cross-role review.

### 6. Address-prompt selection

`_build_address_prompt()` must support architect-authored design revisions.

Add support for:

- `artifact_type="arch_design"` => `ARCH_ADDRESS_DESIGN_COMMENTS`

Existing mappings remain unchanged.

`_review_artifact_type()` should also map:

- `arch_design` => `design_doc`

This lets architect-led design revisions persist as standard design artifacts.

### 7. Report generation

`_generate_report()` needs two new internal branches.

#### Architect-design report

Concise contents:

- requirement
- design review status and iteration count
- final design excerpt / summary
- user approval status
- final workflow outcome
- blocked/failure summary when applicable

#### Testplan-design report

Concise contents:

- requirement
- test-plan review status and iteration count
- final plan excerpt / summary
- user approval status
- final workflow outcome
- blocked/failure summary when applicable

The report format should remain lightweight and consistent with the current
mode-specific report helpers.

### 8. Agent prompt strategy

#### Why prompts matter

The new flows depend on agents emitting the correct delegate block and, in the
cross-role review case, bringing their own domain perspective when given a
role-neutral review prompt.

#### Required prompt changes

`src/myswat/cli/init_cmd.py`

- extend `ARCHITECT_SYSTEM_PROMPT` to describe `MODE: design`
- add `DEVELOPER_SYSTEM_PROMPT`
- add `QA_MAIN_SYSTEM_PROMPT`
- add `QA_VICE_SYSTEM_PROMPT`

#### Important implementation detail for existing projects

Changing only `init_cmd.py` is **not sufficient** because existing projects keep
agent system prompts in the `agents` table.

Therefore the implementation must also add a **data migration** to backfill or
upgrade default prompts for existing agents.

Recommended migration behavior:

- if `developer.system_prompt IS NULL`, set developer default prompt
- if `qa_main.system_prompt IS NULL`, set QA main default prompt
- if `qa_vice.system_prompt IS NULL`, set QA vice default prompt
- if `architect.system_prompt IS NULL`, set new architect default prompt
- if `architect.system_prompt` exactly matches the old v004 default prompt,
  replace it with the new prompt containing `MODE: design`

This preserves user-customized prompts while making the feature work on already
initialized projects such as `myswat`.

### 9. Status and display updates

`src/myswat/cli/main.py`

`_infer_stage_labels()` should recognize the new proposer/reviewer directions.

New label rules:

- `architect -> developer` or `architect -> qa_*` => `Architect Design Review`
- `qa_* -> architect` or `qa_* -> developer` => `Test Plan Review`
- existing developer -> QA sequencing stays unchanged

This keeps `myswat task --details` readable without changing the review-cycle
schema.

## API and Interface Design

### New / changed functions

#### `src/myswat/cli/chat_cmd.py`

```python
def _extract_delegation(text: str) -> tuple[str, str] | None

def _run_design_review_interactive(..., proposer_sm: SessionManager, ...) -> None

def _run_testplan_review_interactive(..., proposer_sm: SessionManager, ...) -> None
```

Chat dispatch behavior:

- keep current `/review` and `/work` commands unchanged
- update automatic post-response delegation handling to dispatch by mode
- do not close the proposer chat session for `design` or `testplan`
- keep existing close-and-reopen behavior for `code`

#### `src/myswat/agents/session_manager.py`

```python
def fork_for_work_item(self, work_item_id: int, purpose: str | None = None) -> SessionManager
```

#### `src/myswat/workflow/engine.py`

```python
class WorkMode(StrEnum):
    architect_design = "architect_design"
    testplan_design = "testplan_design"

@dataclass
class WorkflowResult:
    blocked: bool = False
    failure_summary: str = ""

class WorkflowEngine:
    def __init__(..., arch_sm: SessionManager | None = None) -> None
    def _run_architect_design_mode(self, requirement: str, result: WorkflowResult) -> WorkflowResult
    def _run_testplan_design_mode(self, requirement: str, result: WorkflowResult) -> WorkflowResult
    def _build_review_prompt(..., reviewer: SessionManager | None = None) -> str
    def _run_review_loop(..., abort_on_agent_failure: bool = False) -> tuple[str, int, bool]
```

### Work-item metadata

No new table is required. New metadata usage:

- architect-led design flow => `metadata_json.work_mode = "architect_design"`
- QA-led test-plan flow => `metadata_json.work_mode = "testplan_design"`

This is sufficient for status display and future filtering.

## Data Model Changes

### Schema changes

No new tables are required.

### Data migration required

A new migration is required to backfill multi-role default system prompts for
existing agents.

Reason:

- architect delegation guidance is currently seeded only by v004 / init-time
  defaults
- developer and QA agents currently have no default system prompts
- chat-triggered review quality and delegate emission depend on those prompts

### In-memory model changes

The in-memory `WorkflowResult` dataclass should gain:

- `blocked: bool = False`
- `failure_summary: str = ""`

This is not a schema change; it is used to distinguish execution failure from
normal review failure when the chat wrappers set final work-item status.

### Existing schema reused as-is

The following existing fields already support the feature and require no schema
changes:

- `sessions.parent_session_id`
- `sessions.work_item_id`
- `artifacts.artifact_type`
- `review_cycles.proposal_session_id`
- `review_cycles.review_session_id`
- `work_items.metadata_json.task_state`

## Key Design Decisions and Trade-Offs

### Decision 1: extend delegate blocks instead of adding new commands

**Chosen**: extend `````delegate````` with `MODE:`.

**Why**:

- keeps the architect/QA experience conversational
- preserves current delegation UX
- avoids new public command surface

**Trade-off**:

- requires prompt engineering and parser validation
- invalid `MODE:` values must be handled safely

### Decision 2: reuse `WorkflowEngine` instead of duplicating review logic

**Chosen**: add internal engine modes.

**Why**:

- preserves one persistence and review contract
- minimizes orchestration drift
- reuses cancellation, report, and task-state behavior

**Trade-off**:

- `WorkflowEngine` becomes more mode-aware
- startup ownership and report branching must be generalized carefully

### Decision 3: fork proposer TiDB session and reuse runner

**Chosen**: new session row, shared runner.

**Why**:

- work-item history stays correct
- chat session history is not mutated
- proposer retains in-process conversational context

**Trade-off**:

- runner continuity is strongest in-process, not fully durable across immediate
  post-workflow process exit

### Decision 4: use role-neutral review prompts only where needed

**Chosen**: keep existing prompts for existing flows; add neutral prompts only
for new cross-role review paths.

**Why**:

- minimizes regression risk in full workflow
- avoids rewriting prompt behavior for already-tested flows

**Trade-off**:

- prompt selection becomes slightly more complex
- artifact-type alone is no longer enough to choose a prompt

### Decision 5: backfill prompts via migration, not init only

**Chosen**: ship a migration plus updated init defaults.

**Why**:

- existing projects must work immediately
- current project state is stored in TiDB, not regenerated from repo constants

**Trade-off**:

- requires careful “only replace defaults” logic to avoid overwriting
  user-customized prompts

### Decision 6: treat reviewer/proposer execution failure as `blocked`

**Chosen**: for the new chat-triggered modes, agent execution failures block the
workflow instead of being treated like ordinary review comments.

**Why**:

- a non-zero reviewer exit is not actionable feedback
- it prevents meaningless “please revise” loops
- it matches operator expectations when tooling or infrastructure fails

**Trade-off**:

- introduces one more review-loop option and result-state branch
- wrappers must inspect `result.blocked` in addition to `result.success`

## Dependencies

### Internal dependencies

- `src/myswat/cli/chat_cmd.py`
- `src/myswat/agents/session_manager.py`
- `src/myswat/workflow/engine.py`
- `src/myswat/workflow/prompts.py`
- `src/myswat/cli/main.py`
- `src/myswat/cli/init_cmd.py`
- new migration module registered in `src/myswat/db/schema.py`

### Runtime dependencies

No new external packages are required.

The feature depends on the existing availability of:

- TiDB connectivity for work-item/session persistence
- configured agent roles (`architect`, `developer`, `qa_main`, optionally
  `qa_vice`)
- functioning AI CLIs for the involved agents

## Risks and Mitigations

### Risk: missing required agents

- **Scenario**: QA triggers testplan flow but architect is missing.
- **Mitigation**: validate required roles before creating the work item; print a
  clear error and do not start the workflow.

### Risk: malformed or unsupported delegate mode

- **Scenario**: model emits `MODE: test_plan` or `MODE: review`.
- **Mitigation**: validate against allow-list; warn and ignore instead of
  misrouting.

### Risk: prompt backfill overwrites customized prompts

- **Scenario**: user already customized architect or QA instructions.
- **Mitigation**: migration updates only `NULL` prompts or prompts that exactly
  match the old shipped default.

### Risk: shared-runner continuity not fully durable across immediate exit

- **Scenario**: proposer finishes workflow and quits chat before another turn.
- **Mitigation**: accept in v1; document follow-up option to persist runner
  session IDs at the `sessions` row level if later required.

### Risk: regressions in existing full workflow review prompts

- **Scenario**: review prompt routing changes unintentionally alter current
  design or test-plan stages.
- **Mitigation**: keep old prompt mapping for existing modes and add targeted
  tests proving backward compatibility.

### Risk: reviewer CLI exits non-zero during a chat workflow

- **Scenario**: reviewer returns exit code 1 and no usable verdict.
- **Mitigation**: for new chat-triggered modes, mark the workflow `blocked`,
  persist the failure summary, and stop without asking the proposer to revise.

## Testing Strategy

### 1. Unit tests

#### Delegation parsing

Add/update tests in `tests/test_cli/test_cli_helpers.py`:

- `MODE: design` parses correctly
- `MODE: testplan` parses correctly
- missing `MODE:` defaults to `code`
- mixed-case `mode:` is accepted
- unknown mode is surfaced to dispatcher behavior
- fallback-to-block-content still works when `TASK:` is missing

#### Stage label inference

Add tests for:

- `architect -> developer` => `Architect Design Review`
- `architect -> qa_main` => `Architect Design Review`
- `qa_main -> architect` => `Test Plan Review`
- `qa_vice -> architect` => `Test Plan Review`

#### Session forking

Add tests in `tests/test_agents` or a new targeted test module:

- `fork_for_work_item()` creates a new TiDB session row
- child session carries `work_item_id`
- child session sets `parent_session_id` to the original session
- child session shares the same runner instance
- closing the fork does not clear or replace the original session manager

### 2. Chat command tests

Update `tests/test_cli/test_chat_cmd.py`:

- architect `MODE: design` dispatches to `_run_design_review_interactive()`
- QA `MODE: testplan` dispatches to `_run_testplan_review_interactive()`
- architect `MODE: code` keeps existing inline-review behavior
- role/mode mismatch prints warning and does not run workflow
- chat session is not closed/reopened for design/testplan delegation paths
- proposer session is still reusable after the workflow returns

### 3. Workflow engine tests

Update `tests/test_workflow/test_engine.py` and
`tests/test_workflow/test_engine_extended.py`:

- dispatch routes `architect_design` and `testplan_design`
- constructor fails if those modes are used without `arch_sm`
- startup state is owned by the correct proposer role for each mode
- architect-design happy path
- architect-design review failure
- architect-design reviewer exit failure => `blocked=True`
- architect-design user rejection
- architect-design cancellation
- testplan-design happy path
- testplan-design review failure
- testplan-design reviewer exit failure => `blocked=True`
- testplan-design user rejection
- reviewer-aware prompt selection for developer and architect reviewers
- report generation branches for the two new modes
- `abort_on_agent_failure=False` preserves current behavior in existing flows

### 4. Migration and init tests

Add/update tests for:

- new migration is registered in `src/myswat/db/schema.py`
- migration sets developer/QA prompts when `NULL`
- migration upgrades architect prompt when it matches the old default
- migration leaves customized prompts untouched
- `run_init()` seeds the new default system prompts for fresh projects

### 5. Persistence tests

Add targeted store/engine assertions for:

- work item metadata contains `work_mode`
- architect-led design artifacts persist as `design_doc`
- review cycles record correct proposer/reviewer roles and session IDs
- blocked workflows persist failure summaries and `*_blocked` stages
- `myswat task --details` can infer the intended stage label from review cycles

### 6. Recommended local validation tiers

After implementation, validate incrementally with the project’s existing tiers:

- `./.venv/bin/pytest tests/test_cli -q`
- `./.venv/bin/pytest tests/test_agents tests/test_workflow -q`
- `./.venv/bin/pytest tests/test_models tests/test_config -q`
- optionally `./.venv/bin/pytest -q` if TiDB/embedding environment is suitable

## Implementation Plan

1. Extend delegate parsing and dispatch in `chat_cmd`
2. Add `SessionManager.fork_for_work_item()`
3. Add internal engine modes and proposer-aware startup state
4. Add strict agent-failure handling for chat-triggered review flows
5. Add role-neutral prompts and architect address prompt
6. Add chat interactive wrappers for design/testplan workflows
7. Add agent prompt defaults plus migration backfill
8. Update status-stage label inference
9. Add targeted tests, then broader regression runs

## Files Expected to Change

- `CHAT_DESIGN_WORKFLOW_DESIGN.md`
- `src/myswat/cli/chat_cmd.py`
- `src/myswat/agents/session_manager.py`
- `src/myswat/workflow/engine.py`
- `src/myswat/workflow/prompts.py`
- `src/myswat/cli/main.py`
- `src/myswat/cli/init_cmd.py`
- `src/myswat/db/schema.py`
- `src/myswat/db/migrations/v008_chat_workflow_agent_prompts.py` (name illustrative)
- tests under `tests/test_cli`, `tests/test_agents`, and `tests/test_workflow`

## Acceptance Criteria

- Architect can trigger a team design-finalization workflow from chat using a
  `MODE: design` delegate block.
- QA can trigger a team test-plan-finalization workflow from chat using a
  `MODE: testplan` delegate block.
- Proposer workflow turns appear under the created work item.
- Original chat session remains active and resumes after the workflow.
- Review loops persist artifacts and review cycles using existing store APIs.
- Reviewer/proposer CLI failures in the new chat-triggered flows mark the work
  item `blocked` and surface a clear failure summary.
- `myswat task --details` shows readable review labels for the new flows.
- Existing full workflow behavior remains unchanged.
