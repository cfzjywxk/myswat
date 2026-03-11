# Finalized Plan: `myswat work` Modes

## Final Decision

This change adds explicit work modes to `myswat work` while keeping the current
workflow engine and persistence model. The public CLI gains `--design`/`--plan`,
`--development`/`--dev`, and `--test`/`--ga-test`; the default remains the full
workflow when no mode flag is set. The implementation is limited to:

- `src/myswat/workflow/engine.py`
- `src/myswat/cli/main.py`
- `src/myswat/cli/work_cmd.py`

| Mode | CLI selection | Stages run | Success contract |
|---|---|---|---|
| `full` | default | design -> design review -> plan -> plan review -> development -> GA test -> report | unchanged: all phases committed and GA passed |
| `design` | `--design`, `--plan` | design -> design review -> plan -> plan review -> report | both design and plan reviews passed, and the user did not stop at a checkpoint |
| `development` | `--development`, `--dev` | phased development -> report | all phases committed |
| `test` | `--test`, `--ga-test` | GA test plan/review -> execute tests -> report | GA passed |

## Rationale

- Reuse the existing stage helpers instead of introducing a second workflow stack.
- Keep CLI mode resolution and detached-worker threading in the CLI layer; keep
  stage composition and reporting in `WorkflowEngine`.
- Persist the selected mode in work-item metadata so foreground and background
  runs remain observable without a schema change.

## Invariants

- `WorkflowEngine.run()` remains the shared task-state and process-log envelope;
  mode-specific execution happens behind a dispatcher, not in duplicated
  top-level orchestration code.
- `_run_review_loop()` returns `(artifact, iterations, passed)` and all six
  current call sites are updated:
  1. design review in full/design mode
  2. plan review in full/design mode
  3. code review in `_run_phase()`
  4. test-plan review in `_run_ga_test_phase()`
  5. bug-fix design review in `_run_bug_fix_arch_change()`
  6. bug-fix plan review in `_run_bug_fix_arch_change()`
- `--design` is incompatible with `--background`.
- `--design` sets `auto_approve=False`; `full`, `development`, and `test` keep
  `auto_approve=True`.
- `metadata_json.work_mode` is written on work items created by both
  `_run_workflow()` and `_launch_background_work()`.
- Test mode never escalates an `arch_change` finding into
  `_run_bug_fix_arch_change()`.
- Development-mode QA review remains informational; commit behavior does not
  become a hard approval gate.
- Reports are mode-aware and only show sections for stages that actually ran.
- `final_report` is populated on successful completion and on every early return
  or cancellation path.
- Existing full-mode success semantics stay intact: design/plan review pass
  flags are recorded for reporting but do not block full-mode success.

## Non-Goals

- No DB schema or migration changes.
- No changes to `myswat run`, `myswat chat`, `review_loop.py`, or prompt
  templates.
- No new agent roles, no new storage tables, and no new workflow subsystem.
- No change to local-commit behavior or push/publish semantics.
- No attempt to make interactive design checkpoints work in detached/background
  execution.

## Required Model / Dataclass Changes

These changes all stay in `src/myswat/workflow/engine.py`; no Pydantic model or
schema changes are required.

| Type | Required change | Why |
|---|---|---|
| `WorkMode` | Add enum values `full`, `design`, `development`, `test` | Single typed representation across CLI, worker threading, and engine dispatch |
| `WorkflowResult` | Add `design_review_passed: bool = False` and `plan_review_passed: bool = False` | Preserve review outcomes for design/full reporting and design-mode success |
| `PhaseResult` | Add `review_passed: bool = False` | Make development/full reports explicit that QA review is informational |
| `GATestResult` | Add `test_plan_review_passed: bool = False` | Record whether dev actually approved the GA test plan |

## File-by-File Implementation Plan

### `src/myswat/workflow/engine.py`

1. Add `WorkMode` and accept `mode: WorkMode = WorkMode.full` in
   `WorkflowEngine.__init__`; store it as `self._mode`.
2. Keep `run()` as the shared envelope:
   - create the initial `WorkflowResult`
   - persist `workflow_started`
   - append the task request event
   - dispatch to `_run_full()`, `_run_design_mode()`, `_run_development_mode()`,
     or `_run_test_mode()`
   - persist the final task-state summary from the returned result
3. Extract the current `run()` body into `_run_full()` with stage order
   unchanged. Full mode must store `design_review_passed` and
   `plan_review_passed`, but it must continue to rely on user checkpoints rather
   than review pass/fail as the hard gate.
4. Add `_run_design_mode(requirement)`:
   - run stages 1-4 only
   - call `_run_review_loop()` for design and plan reviews and store both
     `*_review_passed` flags
   - keep the user checkpoints active because design mode is the only
     interactive mode
   - generate a mode-aware report
   - set `result.success = design_review_passed and plan_review_passed`
5. Add `_run_development_mode(requirement)`:
   - skip design and plan generation entirely
   - treat `requirement` as the effective design/plan context when calling
     `_run_phase()`
   - parse phases from the requirement text, with the current single-phase
     fallback if parsing produces nothing
   - keep QA review informational; success remains
     `all(p.committed for p in result.phases)`
6. Add `_run_test_mode(requirement)`:
   - call `_run_ga_test_phase(requirement, design=requirement, plan="",
     dev_summary=requirement, allow_arch_fix=False)`
   - skip design, planning, and phased development
   - set `result.success = ga_result.passed`
7. Change `_run_review_loop()` to return `tuple[str, int, bool]` where `passed`
   is `True` only when all reviewers LGTM the artifact before the loop exits.
8. Update all six `_run_review_loop()` call sites to store the new `passed`
   flag:
   - `WorkflowResult.design_review_passed`
   - `WorkflowResult.plan_review_passed`
   - `PhaseResult.review_passed`
   - `GATestResult.test_plan_review_passed`
   - the two nested bug-fix review sites in `_run_bug_fix_arch_change()`
9. Extend `_run_ga_test_phase()` with `allow_arch_fix: bool = True` and thread
   it to `_run_bug_fix()`.
10. Extend `_run_bug_fix()` with `allow_arch_fix: bool = True`; when
    `assessment == "arch_change"` and `allow_arch_fix` is `False`, return an
    unresolved `BugFixResult` with `arch_change=True`, `fixed=False`, and a
    summary directing the user to rerun with `--development` or default full
    mode.
11. Keep test-plan review informational:
    - record `test_plan_review_passed`
    - still execute the latest test plan even if review did not pass
    - surface the unresolved review state in the report
12. Make checkpoint prompt text conditional on `passed` at all interactive
    checkpoints, including the bug-fix sub-workflow. Prompts must not claim
    something was "approved" when the review loop ended by exhaustion rather
    than LGTM.
13. Make `_generate_report()` mode-aware:
    - only emit design/plan sections for `full` and `design`
    - only emit development sections for `full` and `development`
    - only emit GA sections for `full` and `test`
    - annotate design, plan, test-plan, and phase reviews using the new
      pass/fail fields
14. Ensure every early return path in `_run_full()`, `_run_design_mode()`,
    `_run_development_mode()`, `_run_test_mode()`, and
    `_run_bug_fix_arch_change()` sets a meaningful `final_report` before
    returning.

### `src/myswat/cli/main.py`

1. Extend `work()` with mutually exclusive mode flags:
   - `--design`, `--plan`
   - `--development`, `--dev`
   - `--test`, `--ga-test`
2. Resolve those flags into `WorkMode` and pass `mode` to `run_work()`.
3. Reject `--design` combined with `--background` before any work item,
   session, or subprocess is created.
4. Extend the hidden `work-background-worker` command with `--mode`, defaulting
   to `"full"` for backward compatibility, and convert it back to `WorkMode`
   before calling `run_background_work_item()`.
5. Update the status/work-item display in the same file to prefer
   `metadata_json.work_mode` when present, while keeping the current team/solo
   inference as a fallback for legacy items that do not yet carry the new
   metadata.

### `src/myswat/cli/work_cmd.py`

1. Thread `mode: WorkMode = WorkMode.full` through:
   - `run_work()`
   - `_run_workflow()`
   - `_launch_background_work()`
   - `run_background_work_item()`
2. In `_launch_background_work()`, add a defense-in-depth rejection for
   `WorkMode.design`; background workers run with `stdin=subprocess.DEVNULL` and
   cannot satisfy interactive checkpoints.
3. When spawning the detached worker, append `--mode <mode.value>` to the
   subprocess command.
4. When creating a new work item in `_launch_background_work()`, include
   `metadata_json={"work_mode": mode.value}`.
5. When creating a new work item in `_run_workflow()`, include the same
   `metadata_json={"work_mode": mode.value}`.
6. Preserve metadata merging behavior so `work_mode`, `background`, and
   `task_state` continue to coexist in the same `metadata_json` document.
7. Instantiate `WorkflowEngine` with both `mode=mode` and
   `auto_approve=(mode != WorkMode.design)`.

## CLI and Background Threading Details

- Foreground flow:
  - `myswat work ... --design|--development|--test`
  - `main.py::work()` resolves `WorkMode`
  - `work_cmd.py::run_work(..., mode=...)`
  - `work_cmd.py::_run_workflow(..., mode=...)`
  - `WorkflowEngine(..., mode=mode, auto_approve=...)`
- Background flow:
  - `myswat work ... --background --development|--test`
  - `main.py::work()` resolves `WorkMode`
  - `work_cmd.py::_launch_background_work(..., mode=...)`
  - detached subprocess command becomes `work-background-worker ... --mode <value>`
  - `main.py::work_background_worker()` parses `--mode`
  - `work_cmd.py::run_background_work_item(..., mode=...)`
  - `work_cmd.py::_run_workflow(..., background_worker=True, mode=...)`
- Design mode never uses the background flow because it requires interactive
  checkpoints and `input()` would otherwise read from `/dev/null`.

## Testing Plan

Update existing tests; no new test module is required.

- `tests/test_cli/test_main_cmd.py`
  - mode-flag parsing and alias coverage
  - mutual-exclusion error when more than one mode flag is set
  - `--design --background` rejection
  - hidden `work-background-worker --mode` threading/parsing
  - status fallback behavior for items with and without
    `metadata_json.work_mode`
- `tests/test_cli/test_work_cmd.py`
  - `run_work()` passes `mode` through both foreground and background paths
  - `_launch_background_work()` includes `--mode` in the detached subprocess
    command
  - work items created in both `_run_workflow()` and `_launch_background_work()`
    persist `metadata_json.work_mode`
  - design mode is rejected in background at the launcher layer
  - `WorkflowEngine` is constructed with `mode=...` and `auto_approve=False`
    only for design mode
- `tests/test_workflow/test_engine.py`
  - dispatcher coverage for full/design/development/test mode selection
  - mode-specific success criteria
  - design mode keeps checkpoints interactive while other modes remain
    auto-approved
- `tests/test_workflow/test_engine_extended.py`
  - dataclass default-field coverage for the new result fields
  - `_run_review_loop()` 3-tuple behavior
  - propagation of `passed` through all six call sites, including
    `_run_bug_fix_arch_change()`
  - test mode refusing to escalate `arch_change`
  - mode-aware report sections and labels
  - early-return `final_report` coverage across the extracted mode methods and
    sub-workflow
  - development-mode QA review remaining informational rather than a commit gate

## Acceptance Criteria

- `myswat work` with no mode flag behaves as today, except that reports and
  persisted state gain explicit review-pass metadata.
- `myswat work --design` runs only design and plan stages, requires interactive
  checkpoints, and returns success only when both reviews passed.
- `myswat work --development` runs only phased development, does not generate
  design/plan artifacts, and succeeds when all phases commit.
- `myswat work --test` runs only GA testing, does not start architecture-change
  sub-workflows, and reports unresolved `arch_change` findings as follow-up
  work.
- `_run_review_loop()` returns `(artifact, iterations, passed)` and every
  current caller stores or uses `passed` appropriately.
- Reports omit skipped sections instead of showing zero-iteration placeholders,
  and they label review outcomes accurately.
- Background workers receive the selected mode via hidden
  `work-background-worker --mode`.
- New work items created in both foreground and background paths persist
  `metadata_json.work_mode`.
- Existing background/task metadata continues to coexist with the new
  `work_mode` key.
- Legacy work items without `work_mode` still display correctly via fallback
  logic.

## Rollout and Compatibility

- No migration is required; `work_mode` is additive JSON metadata only.
- Hidden worker compatibility is preserved by making `--mode` optional with a
  default of `full`.
- Existing persisted work items, background metadata, and task-state envelopes
  remain valid.
- The change is isolated to `myswat work`; other CLI commands and prompt
  families continue to behave as they do today.
- The implementation should land as one coordinated change across the three
  files and the named test modules, not as partially threaded intermediate
  states.
