# Conversation Persistence — Implementation Plan

Reference: `CONVERSATION_PERSISTENCE_DESIGN.md`

6 phases, ordered by dependency. Each phase is testable once its dependencies land.

---

## Phase 1: Schema & Store Foundation

**Goal**: Database ready, new store methods available. Old deletion methods are
**deprecated but not yet removed** — callers are updated in later phases first.

### 1a. Register v006 migration in schema.py

`db/schema.py` — add `"myswat.db.migrations.v006_flexible_vector_dimension"` to
`MIGRATION_MODULES` (file exists but was never registered).

### 1b. Create v007 migration

New file `db/migrations/v007_conversation_persistence.py`:
```python
VERSION = 7
DESCRIPTION = "Add compacted_at timestamp and turn recency index for conversation persistence"
STATEMENTS = [
    "ALTER TABLE sessions ADD COLUMN compacted_at TIMESTAMP NULL DEFAULT NULL",
    "CREATE INDEX idx_session_turns_recency ON session_turns (created_at DESC, id DESC)",
]
```
Register in `db/schema.py` `MIGRATION_MODULES`.

### 1c. Add new store methods

`memory/store.py`:

- Add `mark_session_fully_compacted(session_id)` — sets `status='compacted'` and
  `compacted_at=NOW()`.
- Add `get_recent_turns_by_project(project_id, per_role_limit=10, exclude_session_id=None)`
  — window function query, `(created_at DESC, id DESC)` ordering, conditional
  `exclude_session_id` predicate, returns grouped-by-role list.
- Add `get_recent_turns_global(project_id, limit=50, role=None)` — flat query across
  all roles (or filtered to one role if `role` is given). The store queries with
  `ORDER BY created_at DESC, id DESC LIMIT %s` (newest first for efficient limiting),
  then **reverses the result in Python** before returning, so the caller receives
  turns in **chronological order** (oldest first). Returns a flat list of dicts with
  `agent_role` on each row. When `role` is not None, adds `AND a.role = %s`. This is
  distinct from the per-role grouped query above.
- Add `gc_compacted_turns(project_id, grace_days=7, keep_recent=50, dry_run=False)` —
  two-step cutoff + delete, `(created_at, id)` tuple comparison, 0-based offset
  (`keep_recent - 1`). When `dry_run=True`, run the cutoff query + a SELECT COUNT
  instead of DELETE. Returns `{turns_deleted: N, sessions_affected: N}` (the sessions
  count comes from `COUNT(DISTINCT s.id)` in the same WHERE clause).

### 1d. Deprecate old deletion methods (do NOT remove yet)

`memory/store.py`:

- Keep `purge_compacted_sessions()` and `delete_archived_session()` for now — they
  still have callers in `session_manager.py` and `memory_cmd.py`.
- They will be removed in Phase 5 after all callers are updated.

### 1e. Update Session model

`models/session.py` — add `compacted_at: datetime | None = None` field to `Session`.

### Test

- Run migration against dev TiDB.
- Unit test `mark_session_fully_compacted()` sets both status and timestamp.
- Unit test `get_recent_turns_by_project()` with multiple roles, verify per-role limit,
  verify `exclude_session_id=None` works, verify deterministic ordering with same-second
  inserts.
- Unit test `get_recent_turns_global()` returns flat chronological list, respects limit.
- Unit test `get_recent_turns_global(role='developer')` filters correctly.
- Unit test `gc_compacted_turns()` with grace period and recency cutoff edge cases.
- Unit test `gc_compacted_turns(dry_run=True)` returns counts without deleting.
- Existing tests still pass (old methods not removed yet).

---

## Phase 2: Compactor Changes

**Goal**: Compaction threshold raised to 50, parser returns `(items, ok)`, failure
path is retryable, `mark_session_fully_compacted()` used everywhere.

**Depends on**: Phase 1 (new store method).

### 2a. Change `parse_compaction_output()` return type

`memory/compactor.py`:

- Return `(items: list, ok: bool)` tuple.
- `([], True)` — valid empty JSON array (AI found nothing).
- `([], False)` — parse failure (no JSON found, malformed).
- `([...], True)` — successful parse with items.

### 2b. Update `compact_session()` — watermark + status rules

`memory/compactor.py`:

- Unpack `(items, ok) = parse_compaction_output(response.content)`.
- **Watermark advancement rules** (currently watermark is only advanced in the
  `mark_compacted=False` path; the `mark_compacted=True` path skips it):
  - If `response.success and ok` → advance watermark to last included turn_index, then:
    - If `mark_compacted=True` AND last included turn_index == last uncompacted
      turn_index in the session (i.e., all turns were processed, no truncation) →
      call `mark_session_fully_compacted()`.
    - If `mark_compacted=True` BUT truncation occurred (last included < last
      uncompacted) → advance watermark only, leave session as `completed`. The
      remaining turns will be compacted on the next `compact_all_pending()` run.
    - If `mark_compacted=False` → leave session active (mid-session).
  - If `not response.success or not ok` → do NOT advance watermark, do NOT change
    session status. The turns remain uncompacted and retryable.
- **Watermark must match the transcript, not all turns.** The current code computes
  `max_turn_index` over all `new_turns`, but the transcript builder truncates at
  100k chars (`compactor.py:145`). If truncation occurs, turns after the cutoff were
  never sent to the AI and must NOT be marked as compacted. Fix: track the last
  `turn_index` actually included in the transcript (the loop that builds `lines`
  already knows when it hits the limit). Advance watermark to that index, not
  `max(t.turn_index for t in new_turns)`. Remaining turns above the new watermark
  will be picked up in the next compaction round.
- This ensures:
  - Successful final compaction (no truncation): watermark covers all turns, session
    marked `compacted` with `compacted_at` set.
  - Successful final compaction (truncated): watermark advanced through processed
    prefix only, session stays `completed`, `compact_all_pending()` retries remainder.
  - Successful mid-session compaction: watermark advanced through processed turns only.
  - Failed compaction (AI error or parse failure): nothing changes, retryable.
- Replace `self._store.mark_session_compacted()` → `self._store.mark_session_fully_compacted()`.
- Short-session branch in `compact_all_pending()` (< 2 turns): mark `compacted` with
  `compacted_at = NOW()` via `mark_session_fully_compacted()`. Although no knowledge
  entry covers these turns, the turns themselves are trivial (0-1 turns) and not worth
  preserving. Leaving them as `completed` would cause `compact_all_pending()` to
  revisit them on every run forever. Marking them `compacted` moves them to a terminal
  state. GC will eventually delete their turns, which is fine — there's nothing
  meaningful to lose.

### 2c. Deprecate `threshold_tokens` (keep backward-compatible until Phase 5)

- `memory/compactor.py` — keep `threshold_tokens` param in `__init__` signature but
  **ignore it** in `should_compact()` (remove the token check). Change
  `threshold_turns` default to `50`. This way existing CLI call sites that pass
  `threshold_tokens=` don't break at import/construction time.
- `config/settings.py` — keep `compaction.threshold_tokens` for now (removed in Phase 5).
- The parameter and setting are fully removed in Phase 5a alongside the CLI call sites.

### 2d. Update `parse_compaction_output()` callers

- `memory/ingester.py:142` — unpack tuple: `items, _ok = parse_compaction_output(...)`.

### 2e. Update tests

- `tests/test_memory/test_compactor.py` — all assertions change from `list` to
  `(list, bool)` tuple.
- `tests/test_memory/test_compactor_full.py` — same.
- Verify `"[]"` → `([], True)`, `"not json"` → `([], False)`, valid items → `([...], True)`.

### Test

- Run compactor unit tests.
- Integration test: `compact_session` with a mock AI that returns garbage → session
  stays `completed`, not `compacted`.
- Integration test: `compact_session` with a mock AI that returns `[]` → session
  marked `compacted` with `compacted_at` set.

---

## Phase 3: Session Manager — Stop Deleting

**Goal**: `close()` and mid-session compaction preserve all raw turns.

**Depends on**: Phase 2 (compactor changes).

### 3a. Rewrite `close()`

`agents/session_manager.py`:

- Remove the `if self._store.get_session(...) ... delete_archived_session()` block.
- Replace `_compact()` with `_compact_final()`:
  - Call `compact_session(..., mark_compacted=True)` (which now uses
    `mark_session_fully_compacted` internally from Phase 2).
  - Do NOT call `delete_compacted_turns()`.
  - Do NOT call `delete_archived_session()`.

### 3b. Fix `_check_mid_session_compaction()`

`agents/session_manager.py`:

- Remove the `deleted = self._store.delete_compacted_turns(self._session.id)` line.
- Keep `self._store.reset_session_token_count(self._session.id)`.
- Update the stderr print to remove mention of "deleted" turns.

### Test

- Integration test: run a session with > 50 turns, verify mid-session compaction
  fires, verify all turns still exist in `session_turns`.
- Integration test: close a session, verify turns and session row still exist.
- Verify `myswat status` still shows the session correctly.

---

## Phase 4: Retriever — Project-Scoped Pre-load

**Goal**: New sessions load recent turns across all roles, watermark ignored.

**Depends on**: Phase 1 (`get_recent_turns_by_project`).

### 4a. Add `_build_cross_role_history()`

`memory/retriever.py`:

- New method that formats the grouped-by-role output from
  `get_recent_turns_by_project()` into a context section.
- Format: `### [role] Recent Turns` header per role, then turns in chronological order.
- Respects token budget (truncate oldest turns first).

### 4b. Replace fallback history with always-on Tier 1

`memory/retriever.py` — in `build_context_for_agent()`:

- Remove the `KNOWLEDGE_SUFFICIENCY_THRESHOLD` gate and the old section 4 (raw
  session history fallback).
- Remove `KNOWLEDGE_SUFFICIENCY_THRESHOLD` constant.
- Add Tier 1 section: always call `get_recent_turns_by_project()`, pass to
  `_build_cross_role_history()`.
- Budget: `int(max_tokens * 0.25)` for Tier 1.

### 4c. Update `_build_current_session_context()`

`memory/retriever.py`:

- Remove the watermark filter: stop reading `compacted_through_turn_index`, stop
  filtering `turns = [t for t in turns if t.turn_index > watermark]`.
- Load all physically-present turns from the session, ordered by `turn_index`.
- Budget-limited as before (walk backwards, select by token budget).

### 4d. Add pre-load hint

`memory/retriever.py` — in `_build_myswat_cli_context()`:

- Append the history access hint text (see design doc section 8).

### Test

- Unit test: `_build_cross_role_history()` with 4 roles, verify output format.
- Integration test: create sessions for multiple roles, start a new session, verify
  context includes turns from all roles.
- Integration test: verify watermark-ignored behavior — compact a session, start a new
  one, verify compacted turns still appear in context.
- Verify pre-load hint text appears in context output.

---

## Phase 5: CLI Changes

**Goal**: Remove `threshold_tokens` from all CLI constructors, remove `memory purge`,
add `myswat gc` and `myswat history`.

**Depends on**: Phases 1-4.

### 5a. Fully remove `threshold_tokens`

- `cli/chat_cmd.py` — remove `threshold_tokens=` from `KnowledgeCompactor(...)`.
- `cli/run_cmd.py` — same.
- `cli/work_cmd.py` — same.
- `cli/memory_cmd.py` — same.
- `memory/compactor.py` — remove the deprecated `threshold_tokens` param from `__init__`.
- `config/settings.py` — remove `compaction.threshold_tokens` from settings model.

### 5b. Remove `memory purge` command

`cli/memory_cmd.py`:

- Remove the `purge` command function and its registration.

### 5c. Add `myswat gc` command

`cli/main.py` (or new `cli/gc_cmd.py`):

- `myswat gc -p <slug> [--grace-days 7] [--keep-recent 50] [--dry-run]`
- Calls `store.gc_compacted_turns(dry_run=True/False)`.
- `--dry-run` calls with `dry_run=True` — store returns counts without deleting.
- Print summary from the returned dict:
  `Would delete N turns from M sessions` (dry-run) or
  `Deleted N turns from M sessions` (real run).

### 5d. Add `myswat history` command

`cli/main.py` (or new `cli/history_cmd.py`):

- `myswat history -p <slug> [--turns 50] [--role <role>]`
- Calls `store.get_recent_turns_global(project_id, limit=turns, role=role)` — the
  **flat chronological** query (not the per-role grouped Tier 1 query). `--turns`
  controls the total number of turns returned across all roles. `--role` is passed
  through to the store method which filters by agent role.
- Prints turns in chronological order, each prefixed with `[role] [timestamp]`.
- This is what the pre-load hint tells agents to use.

### 5e. Remove old deletion methods and their callers

Now that all callers are updated (Phase 3 removed session_manager calls, 5b removed
purge command):

`memory/store.py`:
- Remove `purge_compacted_sessions()`.
- Remove `delete_archived_session()`.

### Test

- `myswat gc --dry-run` on a project with compacted sessions — verify correct counts.
- `myswat gc` — verify turns deleted, session rows preserved, counts match dry-run.
- `myswat history -p <slug>` — verify flat chronological output across all roles.
- `myswat history -p <slug> --turns 10` — verify limit works.
- `myswat history -p <slug> --role developer` — verify role filter.
- Verify `myswat memory purge` is gone (command not found).
- Verify all CLI entry points construct `KnowledgeCompactor` without `threshold_tokens`.
- Verify `purge_compacted_sessions` and `delete_archived_session` are gone from store.

---

## Phase 6: Test Cleanup

**Goal**: All existing tests pass with the new behavior.

**Depends on**: Phases 1-5.

### 6a. Store tests

- Remove tests for `purge_compacted_sessions()` and `delete_archived_session()` (methods
  removed in Phase 5e).
- Update any tests that assert turns are deleted after compaction.
- Add tests for `gc_compacted_turns()` edge cases (grace period, recency cutoff,
  same-second ordering, fewer than `keep_recent` turns, dry-run mode).
- Add tests for `get_recent_turns_global()` (flat chronological, limit, role filter).

### 6b. Session manager tests

- Update tests that assert `close()` deletes sessions/turns.
- Update tests that assert mid-session compaction deletes turns.
- Add tests verifying turns persist after close and mid-session compaction.

### 6c. Retriever tests

- Update tests that check `KNOWLEDGE_SUFFICIENCY_THRESHOLD` behavior.
- Update tests that verify watermark filtering in `_build_current_session_context()`.
- Add tests for cross-role history loading.

### 6d. CLI tests

- Remove `memory purge` command tests.
- Add `gc` command tests.
- Add `history` command tests.
- Update any tests that pass `threshold_tokens` to `KnowledgeCompactor`.

### Test

- `pytest` — full suite green.

---

## Dependency Graph

```
Phase 1 (Schema + Store)
  │
  ├──→ Phase 2 (Compactor)
  │       │
  │       └──→ Phase 3 (Session Manager)
  │
  └──→ Phase 4 (Retriever)
          │
          └──→ Phase 5 (CLI)
                  │
                  └──→ Phase 6 (Test Cleanup)
```

Phases 2 and 4 can run in parallel after Phase 1.
Phase 3 depends on Phase 2.
Phase 5 depends on Phases 2+4.
Phase 6 is last — clean up after everything lands.
