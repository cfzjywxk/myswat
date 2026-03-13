# Conversation Persistence Technical Design

Status: Proposed

Reference inputs:
- `CONVERSATION_PERSISTENCE_DESIGN.md`
- `CONVERSATION_PERSISTENCE_PLAN.md`

## 1. Overview

This design implements durable conversation persistence for MySwat so newly started AI
sessions can recover project context quickly without replaying entire historical
transcripts. The central change is to separate three concerns that are currently coupled:

1. **Persistence** — every turn is stored in `session_turns` and remains available after
   mid-session compaction and after session close.
2. **Compaction** — older turns are distilled into project-scoped knowledge entries and a
   session watermark records what has already been processed by the compactor.
3. **Garbage collection** — raw turns are deleted later, by explicit operator action,
   subject to grace-period and recency-retention rules.

The design keeps the current MySwat architecture intact:
- `SessionManager` still owns session lifecycle and turn persistence.
- `KnowledgeCompactor` still owns transcript distillation.
- `MemoryRetriever` still owns prompt context assembly.
- `MemoryStore` remains the main persistence boundary.
- the Typer CLI remains the operator surface.

The implementation changes behavior, not ownership. That keeps the patch local to the
existing session/memory stack and avoids introducing a second conversation store or a
parallel retrieval path.

## 2. Problem Statement

Today, MySwat compaction and cleanup are effectively the same operation:
- mid-session compaction advances the watermark and deletes raw turns
- final compaction marks the session compacted and deletes the session row entirely

This creates four product problems:

1. **Context loss across sessions** — a new developer or QA session cannot preload recent
   cross-role discussion because the raw turns may already be gone.
2. **Incorrect watermark semantics** — the watermark is used both as a “processed by
   compactor” marker and as a “hide these turns from retrieval” flag.
3. **Operational fragility** — deleting session rows conflicts with foreign-key usage in
   `knowledge` and `review_cycles`.
4. **Weak inspectability** — agents can search knowledge, but they cannot reliably pull a
   bounded, recent raw history from the project.

## 3. Goals and Non-Goals

### Goals

- Persist all turns until an explicit GC pass removes eligible raw rows.
- Preload recent project history across all roles on first-turn context build.
- Guarantee that the most recent 50 raw turns in a project remain queryable.
- Keep watermark semantics strictly about compaction progress, not retrieval visibility.
- Preserve current compaction-based knowledge distillation and retry behavior.
- Add operator-facing history and GC commands that align with the new storage model.

### Non-Goals

- Rebuilding the session model or replacing TiDB as the backing store.
- Adding turn-level deduplication between raw history and compacted knowledge.
- Changing work-item orchestration or workflow state contracts.
- Introducing background or automatic GC in this phase.
- Removing legacy `archived` enum values from the database schema.

## 4. Architecture Overview and Approach

### 4.1 High-level architecture

The design is a four-stage pipeline:

1. **Write path**
   - `SessionManager.send()` appends user and assistant turns to `session_turns`.
   - No turn is deleted on normal write or close.

2. **Compaction path**
   - `KnowledgeCompactor.compact_session()` processes only turns above the session’s
     `compacted_through_turn_index` watermark.
   - It stores distilled knowledge in `knowledge` and advances the watermark only for
     transcript lines actually sent to the compactor.
   - A session becomes terminally `compacted` only when all relevant turns have been
     covered successfully, or when the session is trivial and intentionally skipped.

3. **Retrieval path**
   - `MemoryRetriever.build_context_for_agent()` always loads:
     - deterministic project/task context
     - recent project-scoped raw turns per role
     - semantic knowledge hits
     - current-session turns
   - Raw-turn visibility depends only on whether rows still exist in `session_turns`.

4. **Cleanup path**
   - `myswat gc` calls `MemoryStore.gc_compacted_turns()`.
   - GC deletes only old raw turn rows from fully compacted sessions, never session rows.
   - GC preserves the latest 50 turns project-wide regardless of session age.

### 4.2 Component responsibilities

| Component | Responsibility in new design |
|---|---|
| `src/myswat/agents/session_manager.py` | Persist turns, trigger compaction, close sessions without deleting rows |
| `src/myswat/memory/compactor.py` | Compact only uncompacted turns, distinguish parse failure from valid empty output, decide when a session is fully compacted |
| `src/myswat/memory/retriever.py` | Build first-turn context with always-on cross-role recent history |
| `src/myswat/memory/store.py` | Expose project-scoped turn queries, full-compaction marking, and GC |
| `src/myswat/db/schema.py` + migrations | Register missing v006 and add v007 storage/index changes |
| `src/myswat/cli/main.py` | Add top-level `history` and `gc` command surface |
| `src/myswat/cli/memory_cmd.py` | Keep `memory compact`; remove obsolete `memory purge` |

### 4.3 End-to-end flow

#### New session start

1. `SessionManager.send()` creates or resumes a TiDB session.
2. On the first AI turn, `MemoryRetriever.build_context_for_agent()` assembles context.
3. Retriever includes 10 most recent turns per role across the project, excluding the
   current session if requested.
4. Retriever also includes knowledge hits and current-session continuity.
5. The AI starts with a bounded but representative view of recent cross-role progress.

#### Mid-session compaction

1. `KnowledgeCompactor.should_compact()` fires at 50 uncompacted turns.
2. `compact_session(..., mark_compacted=False)` extracts knowledge from turns above the
   watermark.
3. On successful parse, watermark advances through the processed prefix only.
4. Session remains `active`; raw turns stay in storage.
5. `SessionManager` resets `token_count_est` only to avoid re-trigger loops.

#### Session close

1. `SessionManager.close()` marks the session `completed`.
2. Final compaction runs if a compactor exists.
3. If all remaining uncompacted turns are covered successfully, the session becomes
   `compacted` and `compacted_at` is set.
4. If compaction fails or transcript truncation leaves work outstanding, the session
   remains `completed` for later retry.
5. No session row or turn row is deleted.

#### Manual garbage collection

1. Operator runs `myswat gc -p <slug>`.
2. Store finds the 50th most recent turn for the project using deterministic
   `(created_at DESC, id DESC)` ordering.
3. Store deletes only turns older than that cutoff from sessions whose status is
   `compacted` and whose `compacted_at` is older than the grace threshold.
4. Session rows remain for provenance and foreign-key integrity.

## 5. Key Design Decisions and Trade-offs

### 5.1 Watermark controls re-compaction, not visibility

**Decision**
- Keep `compacted_through_turn_index` as a compaction-progress marker only.

**Why**
- The same session may need to remain preloadable even after close-time compaction.
- Using the watermark as a visibility filter makes “successful compaction” equivalent to
  “erase the recent conversation from retrieval,” which directly conflicts with the
  requirement.

**Trade-off**
- Some overlap remains between raw turns and compacted knowledge.
- This is accepted because it preserves correctness and keeps the retriever simple.

### 5.2 Preserve raw turns until lazy GC

**Decision**
- Remove turn deletion from both mid-session and close-time flows.

**Why**
- Retrieval and inspectability need physically present turns.
- Deferred GC is safer than eager deletion because it decouples correctness from storage
  optimization.

**Trade-off**
- Storage grows faster between compaction and GC.
- The mitigating controls are a 50-turn always-keep floor, grace-period GC, and a manual
  operator command.

### 5.3 Keep session rows forever

**Decision**
- Never delete `sessions` rows in GC.

**Why**
- `knowledge.source_session_id`, `review_cycles.proposal_session_id`, and
  `review_cycles.review_session_id` all reference `sessions.id`.
- Keeping lightweight session rows is much cheaper than reworking foreign keys or
  provenance semantics.

**Trade-off**
- Session metadata grows monotonically.
- This is acceptable because the large payload lives in `session_turns`, not in
  `sessions`.

### 5.4 Always preload recent raw history

**Decision**
- Replace the current “knowledge sparse only” fallback with an always-on Tier 1 section.

**Why**
- Recent collaboration is operational context, not just long-term knowledge.
- A mature project still needs to expose the latest back-and-forth across architect,
  developer, and QA roles.

**Trade-off**
- The prompt always spends budget on recent history.
- The design caps this at roughly 25% of prompt context and lets older turns be pulled
  on demand via CLI.

### 5.5 Dynamic role discovery instead of hardcoded roles

**Decision**
- Discover roles from `agents` joined through sessions and turns.

**Why**
- The project model already supports custom roles.
- Hardcoding roles would drift from actual project configuration and create CLI/retriever
  blind spots.

**Trade-off**
- The number of groups may vary by project.
- The retriever must enforce a token budget rather than relying only on a fixed row cap.

### 5.6 Distinguish valid empty extraction from parse failure

**Decision**
- Change `parse_compaction_output()` to return `(items, ok)` instead of only `items`.

**Why**
- `[]` can mean either “nothing useful found” or “compactor output was unusable.”
- Those outcomes have different retry semantics.

**Trade-off**
- One extra compatibility touchpoint exists in `src/myswat/memory/ingester.py`.
- The benefit is correct compaction state and safer retries.

### 5.7 Keep `archived` as a legacy enum value

**Decision**
- Stop writing `archived`, but do not change the existing enum in this effort.

**Why**
- The current schema already includes `archived` in `sessions.status`.
- Removing enum values is unrelated to the persistence goal and can be intrusive in TiDB.

**Trade-off**
- The enum remains broader than the active behavioral model.
- This is an acceptable compatibility compromise.

## 6. API and Interface Design

### 6.1 Database and model interface

#### Migration ordering

- Register existing `v006_flexible_vector_dimension` in `src/myswat/db/schema.py`.
- Add `v007_conversation_persistence` after v006.

#### Session model

`src/myswat/models/session.py`

- Add `compacted_at: datetime | None = None`.
- Keep `status` string-compatible with existing stored enum values.

### 6.2 `MemoryStore` additions

`src/myswat/memory/store.py`

#### `mark_session_fully_compacted(session_id: int) -> None`

Behavior:
- sets `status = 'compacted'`
- sets `compacted_at = NOW()`

This replaces “mark compacted” where full terminal compaction is intended.

#### `get_recent_turns_by_project(project_id: int, per_role_limit: int = 10, exclude_session_id: int | None = None) -> list[dict]`

Purpose:
- return grouped recent raw turns across all roles in the project

Contract:
- groups by `agent_role`
- limits rows per role in SQL with `ROW_NUMBER()`
- orders recency with `created_at DESC, id DESC`
- returns each role group in chronological order for prompt assembly
- ignores watermark entirely
- omits `exclude_session_id` predicate when the argument is `None`

Suggested return shape:

```python
[
    {
        "agent_role": "developer",
        "turns": [
            {"role": "user", "content": "...", "created_at": ...},
            {"role": "assistant", "content": "...", "created_at": ...},
        ],
    },
]
```

#### `get_recent_turns_global(project_id: int, limit: int = 50, role: str | None = None) -> list[dict]`

Purpose:
- support the operator/agent-facing `myswat history` command

Contract:
- fetches newest-first for efficient limiting
- reverses in Python before returning so callers receive chronological order
- includes `agent_role` on each row
- optionally narrows by role

Suggested row shape:

```python
{
    "session_id": 123,
    "agent_role": "qa_main",
    "role": "assistant",
    "content": "...",
    "created_at": ...,
    "turn_index": 8,
}
```

#### `gc_compacted_turns(project_id: int, grace_days: int = 7, keep_recent: int = 50, dry_run: bool = False) -> dict`

Purpose:
- reclaim raw-turn storage while preserving a global recent-history floor

Contract:
- operates only on project turns
- deletes only rows in `session_turns`
- considers only sessions with `status = 'compacted'`
- requires `compacted_at < NOW() - INTERVAL grace_days DAY`
- preserves all turns at or above the recency cutoff
- supports `dry_run`
- returns:

```python
{
    "turns_deleted": 123,
    "sessions_affected": 7,
}
```

Notes:
- if the project has fewer than `keep_recent` turns total, GC is a no-op
- the cutoff must use `(created_at, id)` to stay deterministic for same-second inserts

### 6.3 `KnowledgeCompactor` changes

`src/myswat/memory/compactor.py`

#### Constructor

```python
KnowledgeCompactor(
    store: MemoryStore,
    runner: AgentRunner | None = None,
    threshold_turns: int = 50,
)
```

- remove `threshold_tokens` from the public constructor in the final interface
- temporarily tolerate phased rollout if needed, but the implementation end-state should
  remove the parameter from all call sites and settings

#### `parse_compaction_output(raw_output: str) -> tuple[list[dict[str, Any]], bool]`

Return rules:
- `([], True)` — valid empty JSON array
- `([], False)` — no parseable JSON array / malformed output
- `([items...], True)` — parsed successfully

#### `should_compact(session_id: int) -> bool`

- compact based only on uncompacted turn count
- threshold is `>= 50`
- do not use token-count triggers

#### `compact_session(..., mark_compacted: bool = True) -> list[int]`

Behavioral rules:
- process only turns above watermark
- compute the **last included** `turn_index` based on transcript truncation, not by max
  turn in the candidate set
- on runner failure or parse failure: do not advance watermark and do not change status
- on success with parseable output: advance watermark through the processed prefix
- if `mark_compacted=True` and the processed prefix covers all remaining uncompacted
  turns: call `mark_session_fully_compacted()`
- if truncation leaves uncompacted suffix turns: leave session `completed`

### 6.4 `SessionManager` changes

`src/myswat/agents/session_manager.py`

#### `close()`

- continue to mark the session `completed`
- run final compaction if a compactor exists
- never call `delete_archived_session()`
- never delete turn rows directly

#### `_compact_final()`

- new helper wrapping final compaction behavior
- centralizes final-session watermark/status logic
- isolates “close semantics” from “mid-session semantics”

#### `_check_mid_session_compaction()`

- keep the compaction trigger
- remove `delete_compacted_turns()` call
- keep `reset_session_token_count()`
- update stderr summary to report only created knowledge entries

### 6.5 Retriever interface

`src/myswat/memory/retriever.py`

#### Context assembly order

1. Project access instructions / history hint
2. Project operations knowledge
3. Current work-item state
4. Tier 1 recent cross-role raw turns
5. Tier 2 searched compacted knowledge
6. Tier 3 current-session turns

This order intentionally exposes recent operational flow before semantic summaries.

#### `_build_cross_role_history(recent_turns: list[dict], budget_tokens: int) -> str`

Purpose:
- format grouped recent turns into prompt-safe markdown

Contract:
- emits `### [role] Recent Turns` headers
- prints turns in chronological order per role
- truncates oldest material first when over budget

#### `_build_current_session_context(session_id: int, budget_tokens: int) -> str`

Change:
- stop filtering by `compacted_through_turn_index`
- load all physically present turns for the current session

Rationale:
- `/reset` or AI session restart should be able to rebuild from existing raw turns even
  if the session has already been compacted.

### 6.6 CLI surface

Per repository convention, new CLI surface stays wired through `src/myswat/cli/main.py`.

#### New command: `myswat gc`

Proposed interface:

```bash
./myswat gc -p <slug> [--grace-days 7] [--keep-recent 50] [--dry-run]
```

Behavior:
- resolves the project slug
- calls `gc_compacted_turns()`
- prints either “Would delete ...” or “Deleted ...”

#### New command: `myswat history`

Proposed interface:

```bash
./myswat history -p <slug> [--turns 50] [--role <role>]
```

Behavior:
- resolves the project slug
- reads chronological output from `get_recent_turns_global()`
- prints one line per turn, prefixed with role and timestamp

Suggested display format:

```text
[developer] [2026-03-13 10:31:12] user: implement the persistence flow
[developer] [2026-03-13 10:31:28] assistant: I will update the retriever first
```

#### Removed command: `myswat memory purge`

Rationale:
- the old semantics delete sessions and all compacted turns immediately
- that conflicts with the new retention model and FK safety

## 7. Data Model Changes

### 7.1 Schema changes

#### `sessions.compacted_at`

```sql
ALTER TABLE sessions ADD COLUMN compacted_at TIMESTAMP NULL DEFAULT NULL;
```

Semantics:
- null while active or merely completed
- set exactly when the session becomes terminally compacted
- used only for GC eligibility, not retrieval

#### `idx_session_turns_recency`

```sql
CREATE INDEX idx_session_turns_recency ON session_turns (created_at DESC, id DESC);
```

Purpose:
- supports deterministic project-scoped recency lookup
- gives both `get_recent_turns_by_project()` and GC cutoff selection a stable order

### 7.2 Session status semantics

Behavioral model after this change:

| Status | Meaning |
|---|---|
| `active` | Session is open and receiving turns |
| `completed` | Session is closed, but full compaction may still be outstanding or retryable |
| `compacted` | Session is terminal for raw-history lifecycle; `compacted_at` is set |
| `archived` | Legacy enum value; no new writes in this feature |

### 7.3 Retention invariants

The implementation must preserve these invariants:

1. `session_turns` rows are the source of truth for raw history until GC removes them.
2. watermark advancement never outruns the transcript actually sent to the compactor.
3. no GC path deletes `sessions` rows.
4. a project’s most recent 50 raw turns remain queryable.
5. a parse failure never changes session status or watermark.
6. a valid empty extraction may still legitimately complete compaction.

## 8. Dependencies, Compatibility, and Risks

### 8.1 Implementation dependencies

- `v006_flexible_vector_dimension` must be registered before adding v007.
- `MemoryStore.mark_session_fully_compacted()` must land before final compactor logic is
  switched over.
- retriever changes depend on project-scoped turn queries existing in the store.
- CLI `history` depends on the flat global recency query.
- CLI `gc` depends on both `compacted_at` and terminal session-state semantics.

### 8.2 Technical risks

#### Risk: compaction parse ambiguity causes false terminal states

Current risk:
- malformed compactor output collapses to `[]`, which looks like a valid no-op

Mitigation:
- split parse success from extracted content with `(items, ok)`

#### Risk: transcript truncation advances watermark too far

Current risk:
- `max_turn_index` of all candidate turns may exceed the last turn actually present in
  the transcript body

Mitigation:
- track the last included turn while building transcript lines

#### Risk: query cost for per-role recent history

Current risk:
- project-scoped recent-history loading broadens the scan surface compared with
  agent-scoped fallback history

Mitigation:
- add the recency index
- cap results with `ROW_NUMBER()` in SQL
- enforce a token budget in formatting

#### Risk: storage growth before GC

Current risk:
- turn rows are retained longer than before

Mitigation:
- provide explicit `myswat gc`
- preserve only the 50 most recent raw turns project-wide
- rely on session-level compaction to keep older material searchable as knowledge

#### Risk: old tests and call sites assume deletion behavior

Current risk:
- multiple tests and CLI constructors encode the old threshold/deletion model

Mitigation:
- phase the work exactly as described in the plan: schema/store first, then compactor,
  then session manager/retriever, then CLI cleanup

### 8.3 Compatibility notes

- Existing data remains valid; new columns are additive.
- Existing completed sessions with null `compacted_at` will simply not be GC-eligible
  until reprocessed and terminally marked compacted.
- `get_compactable_sessions()` can continue querying `status = 'completed'`.
- `myswat status` remains compatible because sessions continue to exist and retain turns.
- The design intentionally does not alter work-item persistence or knowledge search APIs.

## 9. Testing Strategy

Testing should follow the repository’s existing split between model/store, agent, CLI,
and workflow tests.

### 9.1 Unit tests

#### Store

- `mark_session_fully_compacted()` sets both status and timestamp
- `get_recent_turns_by_project()`
  - enforces per-role limit
  - respects `exclude_session_id`
  - returns grouped chronological output
  - behaves deterministically when timestamps tie
- `get_recent_turns_global()`
  - returns chronological output
  - respects `limit`
  - respects optional role filter
- `gc_compacted_turns()`
  - no-ops when fewer than `keep_recent` turns exist
  - respects grace period
  - preserves recent turns inside otherwise old sessions
  - reports counts correctly in `dry_run`

#### Compactor

- `parse_compaction_output()` returns the correct `(items, ok)` tuple for:
  - clean JSON array
  - fenced code block
  - valid empty array
  - malformed output
- `compact_session()`
  - advances watermark only on success + parse success
  - leaves watermark untouched on runner failure
  - leaves watermark untouched on parse failure
  - marks full compaction only when all remaining turns were processed
  - handles valid empty extraction as a terminal success case
  - respects transcript truncation boundaries

#### Retriever

- always loads cross-role recent history regardless of knowledge count
- ignores watermark in `_build_current_session_context()`
- formats role blocks correctly
- includes the preload hint text

#### Session manager

- `close()` never deletes turns or session rows
- mid-session compaction resets token count but does not delete turns
- stderr messaging no longer claims deletion

### 9.2 Integration tests

- create multi-role sessions, compact one, then start a new session and verify context
  includes recent turns from architect, developer, and QA
- compact a session with same-second inserted turns and verify deterministic history order
- close a session after successful final compaction and verify:
  - session row still exists
  - turns still exist
  - status is `compacted`
  - `compacted_at` is set
- close a session after failed compaction and verify:
  - status remains `completed`
  - turns still exist
  - later `compact_all_pending()` can retry it
- run GC and verify only old turns are deleted while session rows remain

### 9.3 CLI tests

- `myswat gc --dry-run` reports expected counts
- `myswat gc` deletes expected rows and preserves session rows
- `myswat history -p <slug>` prints chronological multi-role history
- `myswat history -p <slug> --turns 10` respects total-turn limit
- `myswat history -p <slug> --role developer` filters correctly
- `myswat memory purge` is no longer available
- all CLI constructors instantiate `KnowledgeCompactor` without `threshold_tokens`

### 9.4 Regression tests

- `myswat status` still shows active and compacted sessions correctly
- `myswat learn` / project-ops retrieval remains unchanged
- `memory/ingester.py` continues to distill or fall back correctly after the parser API
  change
- workflow tests still pass because session persistence changes should be transparent to
  work-item orchestration

### 9.5 Validation sequence

Recommended execution order:

1. targeted memory/store tests
2. targeted compactor tests
3. session manager tests
4. retriever tests
5. CLI tests for `gc` and `history`
6. full workflow/memory suite

## 10. Implementation Sequencing

The existing six-phase plan is sound and should be followed as-is:

1. schema and store foundation
2. compactor behavior changes
3. session-manager deletion removal
4. retriever project-scoped preload
5. CLI cleanup and new commands
6. test cleanup and regression sweep

The only sequencing rule that should not be violated is this: **do not remove old
deletion methods until all callers have been migrated**. That prevents intermediate
breakage while phases land.

## 11. QA Review Checklist

QA should approve implementation only if all of the following are true:

- New sessions can see recent turns from other roles in the same project.
- `compacted_through_turn_index` no longer hides raw turns from retrieval.
- Mid-session compaction never deletes raw turns.
- Closing a session never deletes the session row.
- `compacted_at` is set only for terminally compacted sessions.
- `myswat history` can show up to 50 recent raw turns project-wide.
- `myswat gc` preserves the recent-history floor and never deletes session rows.
- Parse failure leaves sessions retryable.
- Same-second recency ordering is deterministic.

