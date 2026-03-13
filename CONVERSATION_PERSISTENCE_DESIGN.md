# Conversation Persistence & Context Restore Design

## Goal

When a user runs `./myswat chat run -p proj` or `./myswat work "" -p proj`, the newly
created AI agent session should quickly restore context and memory from the previous
project progress — without force-feeding the entire history.

## Core Principles

1. **All conversations are persisted.** Never delete turns on session close.
2. **Recent turns are pre-loaded across all roles.** A new Dev session sees what QA and
   Architect said recently, not just its own history.
3. **50 raw turns are always available.** The most recent 50 turns per project are kept
   raw and queryable. Agents can self-serve deeper history via `myswat status`/`myswat history`.
4. **Compaction converts old turns into knowledge.** Runs mid-session (at 50 uncompacted
   turns) and on session close. Compacted knowledge is loaded like any other knowledge entry.

## Key Design Decision: Watermark ≠ Visibility

The compaction watermark (`compacted_through_turn_index`) controls only **what gets
re-compacted** — it prevents the same turns from being fed to the compaction AI twice.

The watermark does **NOT** control raw turn visibility anywhere:

- **Tier 1 cross-role pre-load**: selects by recency, ignores watermark
- **Current-session restore** (after `/reset` or AI session restart):
  `_build_current_session_context()` must also ignore the watermark and load all
  physically-present turns from the active session by recency

Raw turns remain visible until GC physically deletes them.

This decoupling resolves the contradiction where close-time compaction would advance the
watermark past all turns, making them invisible to pre-load or session restore.

## Data Model

### Existing tables (no changes)

| Table | Role |
|-------|------|
| `sessions` | Tracks session lifecycle, compaction watermark, status |
| `session_turns` | Raw turn storage, ordered by `turn_index` |
| `knowledge` | Compacted knowledge entries, project-scoped |

### Schema changes (migration v007)

**1. Add `compacted_at` to sessions:**

```sql
ALTER TABLE sessions ADD COLUMN compacted_at TIMESTAMP NULL DEFAULT NULL;
```

**Why**: GC needs a stable timestamp for "when was this session fully compacted?" The
existing `updated_at` is updated by progress notes, turn appends, etc. and is unreliable.
`compacted_at` is set exactly once — when the final compaction pass completes (all turns
covered by knowledge). GC uses `compacted_at + grace_days` to decide eligibility.

**2. Add index for project-scoped turn recency queries:**

```sql
CREATE INDEX idx_session_turns_recency ON session_turns (created_at DESC, id DESC);
```

**Why**: The Tier 1 pre-load query joins `session_turns → sessions → agents` filtered by
`project_id`, ordered by `session_turns.created_at DESC`. Without this index, every new
session startup scans all turns in the project. The composite `(created_at DESC, id DESC)`
index serves two purposes: it satisfies the ORDER BY efficiently, and the `id DESC`
tie-breaker ensures deterministic ordering when multiple turns share the same second
(common in multi-agent workflows where turns land concurrently). All recency queries in
this design must use `ORDER BY ... created_at DESC, id DESC` consistently.

Note: v006 (`v006_flexible_vector_dimension.py`, VERSION=6) already exists in the repo.
This migration must be v007.

### Session status model

```
active       → session is in use, turns being appended
completed    → session closed, turns still raw (no compaction yet, or partial)
compacted    → terminal state: compaction complete (compacted_at is set)
               Either all turns are covered by knowledge, or the session is
               trivial (< 2 turns) and has nothing worth extracting.
```

- `close()` sets status to `completed`
- Final compaction (when all turns are covered) sets status to `compacted` AND
  sets `compacted_at = NOW()`
- GC deletes raw turns (not session rows) from `compacted` sessions where
  `compacted_at < NOW() - grace_days`

**Session rows are never deleted by GC.** The `knowledge.source_session_id`,
`review_cycles.proposal_session_id`, and `review_cycles.review_session_id` columns all
have foreign keys to `sessions.id`. Deleting a session row would violate these FKs or
require cascading changes. Session rows are lightweight (no LONGTEXT) — keeping them
permanently is acceptable. GC only deletes `session_turns` rows.

**Compatibility with existing code**:
- `get_compactable_sessions()` queries `status = 'completed'` — still correct, finds
  sessions that were closed but not yet fully compacted
- `purge_compacted_sessions()` — **removed**, replaced by `gc_compacted_turns()`
- `compact_all_pending()` iterates `completed` sessions — still correct

## Context Loading on New Session Start

When `SessionManager.send()` builds context for the first turn of a new AI session,
`MemoryRetriever.build_context_for_agent()` loads:

### Tier 0: Deterministic (always loaded, no search)
- **Project ops** — from `myswat.md` or `knowledge` table (`project_ops` category)
- **Work item state** — current task stage, TODOs, process log
- **MySwat CLI hint** — tells the agent how to query deeper history

### Tier 1: Recent raw turns (pre-loaded, project-scoped)
- **10 most recent turns per role** across the project
- Roles are discovered dynamically from the `agents` table (not hardcoded)
  - Default roles: `architect`, `developer`, `qa_main`, `qa_vice` (4 roles × 10 = 40 max)
  - Custom roles are included automatically
- Scoped to the **project**, not the individual agent — all roles see each other
- Selected by **recency** (`ORDER BY created_at DESC, id DESC`), **ignoring the watermark**
  - A compacted turn is still pre-loadable as long as it physically exists
- Only physically deleted (GC'd) turns are invisible

### Tier 2: Compacted knowledge (searched)
- Knowledge entries from compaction (category: `decision`, `architecture`, `progress`, etc.)
- Searched by vector similarity + keyword matching against the current task description
- These are the distilled summaries of older conversations

**Overlap with Tier 1 / Tier 3 raw turns**: After compaction, the same conversation may
appear as both raw turns (Tier 1 or current-session restore) and compacted knowledge
(Tier 2). This overlap is **accepted, not deduplicated**. Reasons:

- Raw turns and knowledge serve different purposes: raw turns preserve exact wording
  and recent conversational flow; knowledge distills decisions, patterns, and insights.
  The AI agent benefits from both.
- Session-level dedup (excluding all knowledge from a session if any of its raw turns
  are loaded) is too coarse. A long session with mid-session compaction would lose its
  older distilled knowledge just because 10 recent raw turns were loaded.
- Turn-level dedup via `source_turn_ids` (JSON column) is impractical in SQL and adds
  complexity for minimal gain.
- The token budgets for Tier 1 (~2000 tokens) and Tier 2 (~3000 tokens) already cap
  total context consumption. The mild overlap between a few raw turns and a compacted
  summary costs at most a few hundred tokens — well within the budget margin.

No dedup logic is needed in the retriever. Knowledge search runs the same query
regardless of which sessions appear in Tier 1.

### Tier 3: Pre-load hint (pull-based)
- A text block injected into the agent's context:
  ```
  [NOTE] Recent turns are persisted for this project. You are seeing the 10 most
  recent per role. If you need more context, use:
  - `myswat status -p <slug>` — project overview, work items, process log
  - `myswat history -p <slug> --turns 50` — raw recent turns
  ```
- This lets the agent **pull** deeper history on demand, rather than being force-fed

## Turn Persistence Lifecycle

```
Session active
  │
  ├─ All turns saved to session_turns (always)
  │
  ├─ At 50 uncompacted turns → mid-session compaction
  │   ├─ AI extracts knowledge from turns above watermark
  │   ├─ Advance compaction watermark
  │   ├─ Reset token_count_est (avoid re-triggering)
  │   └─ DO NOT delete raw turns (they stay visible for pre-load)
  │
  ├─ Session close
  │   ├─ Mark session status = 'completed'
  │   ├─ If uncompacted turns remain → run compaction, advance watermark
  │   ├─ If ALL turns now below watermark → mark status = 'compacted',
  │   │   set compacted_at = NOW()
  │   └─ Raw turns are NOT deleted
  │
  └─ GC (lazy, separate pass — `myswat gc`)
      ├─ Find the (created_at, id) of the 50th most recent turn in the project
      │   (this is the "recency cutoff" — deterministic across same-second inserts)
      ├─ Find sessions where status = 'compacted'
      │   AND compacted_at < NOW() - grace_days (default 7)
      ├─ For each eligible session:
      │   └─ Delete turns where (created_at, id) < cutoff
      │      (turns at or above the cutoff are kept — they're in the recent 50)
      └─ Session rows are NEVER deleted (FK referenced by knowledge, review_cycles)
```

### What changes from current behavior

| Aspect | Current | New |
|--------|---------|-----|
| `close()` | Compacts → marks compacted → **deletes session+turns** | Compacts → marks completed/compacted → **keeps everything** |
| Mid-session compaction | Compacts → **deletes turns** → resets tokens | Compacts → advances watermark → resets tokens → **keeps turns** |
| History loading | Agent-scoped (`get_recent_history_for_agent`) | **Project-scoped** (all roles' recent turns) |
| Turn visibility | Filtered by watermark | Filtered by **physical existence** (recency-based) |
| Pre-load amount | Knowledge-first, raw history only as fallback | **Always pre-load 10 turns/role** + knowledge |
| Compaction threshold | 10 turns / 5000 tokens | **50 uncompacted turns** (token threshold removed) |
| Raw turn deletion | Immediate after compaction | **Lazy GC** with 7-day grace, separate command |
| Queryable history | Not exposed to agents | **50 turns always queryable** via CLI |
| Role set | Hardcoded 3 roles | **Dynamic** from agents table |

## Implementation Changes

### 1. Schema migration v007

New file `v007_conversation_persistence.py`:

```python
VERSION = 7
DESCRIPTION = "Add compacted_at timestamp and turn recency index for conversation persistence"
STATEMENTS = [
    "ALTER TABLE sessions ADD COLUMN compacted_at TIMESTAMP NULL DEFAULT NULL",
    "CREATE INDEX idx_session_turns_recency ON session_turns (created_at DESC, id DESC)",
]
```

Register in `schema.py` `MIGRATION_MODULES` (after the existing v006 entry, which also
needs to be registered — it exists as a file but is missing from `MIGRATION_MODULES`).

### 2. `SessionManager.close()` — stop deleting

```python
def close(self) -> None:
    if self._session is None:
        return
    self._store.close_session(self._session.id)  # status → 'completed'

    # Run final compaction on remaining uncompacted turns
    if self._compactor:
        self._compact_final()

    # DO NOT delete session or turns — GC handles that later
```

The `_compact_final()` method:
- Compacts uncompacted turns into knowledge (same as current `_compact`)
- Advances the watermark
- Checks if ALL turns are now below watermark:
  - If yes → set status = `compacted`, set `compacted_at = NOW()`
  - If no (e.g. compaction failed for some turns) → leave as `completed`
- Does NOT delete turns or session

### 3. `SessionManager._check_mid_session_compaction()` — stop deleting turns

Remove the `delete_compacted_turns()` call:

```python
def _check_mid_session_compaction(self) -> None:
    if self._session is None or self._compactor is None:
        return
    if not self._compactor.should_compact(self._session.id):
        return
    try:
        ids = self._compactor.compact_session(
            session_id=self._session.id,
            project_id=self._project_id,
            agent_id=self._agent_row["id"],
            mark_compacted=False,
        )
        # Watermark is advanced inside compact_session()
        # Reset token count to avoid re-triggering
        self._store.reset_session_token_count(self._session.id)
        # DO NOT call delete_compacted_turns() — turns stay for pre-load
        ...
```

### 4. `MemoryStore` — add project-scoped turn loading

```python
def get_recent_turns_by_project(
    self, project_id: int,
    per_role_limit: int = 10,
    exclude_session_id: int | None = None,
) -> list[dict]:
    """Fetch recent turns across ALL roles in a project.

    Selects by recency (created_at DESC), NOT by compaction watermark.
    A turn is loadable as long as it physically exists in session_turns.
    Roles are discovered dynamically from the agents table.

    Returns list of {agent_role, turns: [{role, content, created_at}]}
    grouped by agent role, each group sorted chronologically.
    """
```

SQL uses a window function to limit per-role on the DB side, avoiding unbounded scans:

```sql
-- When exclude_session_id is not None:
SELECT role, content, created_at, agent_role FROM (
    SELECT st.role, st.content, st.created_at, st.id AS turn_id,
           a.role AS agent_role,
           ROW_NUMBER() OVER (
               PARTITION BY a.role ORDER BY st.created_at DESC, st.id DESC
           ) AS rn
    FROM session_turns st
    JOIN sessions s ON st.session_id = s.id
    JOIN agents a ON s.agent_id = a.id
    WHERE a.project_id = %s
      AND s.id != %s  -- exclude current session
) ranked
WHERE rn <= %s  -- per_role_limit
ORDER BY agent_role, created_at ASC, turn_id ASC
```

**`exclude_session_id` handling**: The `AND s.id != %s` predicate is only appended when
`exclude_session_id is not None`. When `None`, the WHERE clause omits it entirely (no
session exclusion). This avoids the `s.id != NULL` trap which would filter out everything.

The `ROW_NUMBER()` window function partitions by agent role and ranks by recency. The
outer query filters to the top N per role. This returns at most `N_roles × per_role_limit`
rows regardless of total project history size.

The `idx_session_turns_recency` index (added in v007) helps the ORDER BY inside the
window function. TiDB supports window functions natively.

**No watermark filter.** **No hardcoded role list.** Roles are discovered dynamically
from the join.

### 5. `MemoryStore` — add `mark_session_fully_compacted()`

```python
def mark_session_fully_compacted(self, session_id: int) -> None:
    """Mark a session as fully compacted with timestamp for GC."""
    self._pool.execute(
        "UPDATE sessions SET status = 'compacted', compacted_at = NOW() "
        "WHERE id = %s",
        (session_id,),
    )
```

### 6. `MemoryRetriever.build_context_for_agent()` — always load recent turns

Replace the current fallback-only raw history with always-on project-scoped loading:

```python
# ── Tier 1: Recent turns (ALWAYS loaded, project-scoped) ──
history_budget = int(max_tokens * 0.25)
recent_turns = self._store.get_recent_turns_by_project(
    project_id=project_id,
    per_role_limit=10,
    exclude_session_id=current_session_id,
)
if recent_turns:
    sections.append(self._build_cross_role_history(recent_turns, history_budget))
```

Remove the `KNOWLEDGE_SUFFICIENCY_THRESHOLD` gate — recent turns are always loaded
regardless of knowledge count.

### 7. `KnowledgeCompactor` — raise threshold to 50, remove token trigger

```python
class KnowledgeCompactor:
    def __init__(self, ..., threshold_turns: int = 50):
        # Remove threshold_tokens parameter entirely
```

`should_compact()` becomes:

```python
def should_compact(self, session_id: int) -> bool:
    session = self._store.get_session(session_id)
    if not session or session.get("status") == "compacted":
        return False
    uncompacted = self._store.count_uncompacted_turns(session_id)
    return uncompacted >= self._threshold_turns
```

**Ripple changes for `threshold_tokens` removal**: The `threshold_tokens` parameter is
currently wired through settings and multiple call sites. All must be updated:
- `config/settings.py` — remove `compaction.threshold_tokens` from `MySwatSettings`
- `cli/chat_cmd.py` — remove `threshold_tokens=` from `KnowledgeCompactor(...)` call
- `cli/run_cmd.py` — same
- `cli/work_cmd.py` — same
- `cli/memory_cmd.py` — same (the `memory compact` command)

### 8. Pre-load hint injection

Add to `MemoryRetriever._build_myswat_cli_context()`:

```python
hint = (
    "\n## History Access\n\n"
    "Recent turns are persisted for this project. You are seeing the 10 most "
    "recent per role. If you need more context:\n"
    f"- `{launcher} status -p {slug}` — project overview\n"
    f"- `{launcher} history -p {slug} --turns 50` — raw recent turns\n"
)
```

### 9. Lazy GC (new command: `myswat gc`)

Add `MemoryStore.gc_compacted_turns()`:

```python
def gc_compacted_turns(
    self, project_id: int, grace_days: int = 7, keep_recent: int = 50,
) -> dict:
    """Delete raw turns from fully-compacted sessions that are past the grace period.

    Turns in the most recent `keep_recent` per project are always preserved.
    Session rows are NEVER deleted (FK referenced by knowledge, review_cycles).

    Returns {turns_deleted: N}
    """
```

Implementation (single DELETE with subquery):

```sql
-- Step 1: Find the recency cutoff (the Nth most recent turn's identity)
-- OFFSET is 0-based, so OFFSET (keep_recent - 1) gives the Nth row.
-- Example: keep_recent=50 → OFFSET 49 → returns the 50th most recent turn.
-- If fewer than keep_recent turns exist, this returns no rows → skip GC.
-- Uses (created_at, id) for deterministic ordering across same-second inserts.
SELECT created_at, id FROM (
    SELECT st.created_at, st.id
    FROM session_turns st
    JOIN sessions s ON st.session_id = s.id
    JOIN agents a ON s.agent_id = a.id
    WHERE a.project_id = %s
    ORDER BY st.created_at DESC, st.id DESC
    LIMIT 1 OFFSET %s  -- keep_recent - 1 (0-based)
) AS cutoff_row;

-- Step 2: Delete old turns from GC-eligible sessions
-- A turn is "older than the cutoff" if (created_at, id) < (cutoff_created_at, cutoff_id)
DELETE st FROM session_turns st
JOIN sessions s ON st.session_id = s.id
JOIN agents a ON s.agent_id = a.id
WHERE a.project_id = %s
  AND s.status = 'compacted'
  AND s.compacted_at < NOW() - INTERVAL %s DAY
  AND (st.created_at < %s OR (st.created_at = %s AND st.id < %s));
```

The GC rule is: a turn is deleted when ALL of these are true:
1. Its session is in terminal `compacted` state (compaction complete or trivial session)
2. The session's `compacted_at` is older than `grace_days`
3. The turn's `created_at` is older than the recency cutoff (not in recent 50)

This handles sessions that straddle the recency cutoff correctly: recent turns in an
old session are preserved, old turns in the same session are deleted.

Add `myswat gc -p <slug>` CLI command that calls this.

**`myswat memory purge` is replaced by `myswat gc`.** The existing `purge` command and
`purge_compacted_sessions()` are removed entirely. `myswat gc` is the single user-facing
command for reclaiming storage. It uses `gc_compacted_turns()` which:
- Only deletes `session_turns` rows (never session rows — FK safety)
- Respects the grace period and recency cutoff
- Is safe to run at any time (idempotent)

The old `purge` semantics (delete session rows + all their turns immediately) are
incompatible with the new model and would break FK constraints.

## Context Budget Allocation

For a default `max_tokens = 8000`:

| Tier | Budget | Content |
|------|--------|---------|
| 0 | ~2000 tokens | Project ops + work item state + CLI hint |
| 1 | ~2000 tokens | N × 10 recent raw turns (dynamic role count) |
| 2 | ~3000 tokens | Compacted knowledge (vector search) |
| 3 | ~1000 tokens | Current session turns (within-session continuity) |

Note: with 4 default roles × 10 turns = 40 turns, the Tier 1 budget may need to be
enforced by token count (truncate oldest turns first) rather than strict turn count, to
stay within ~2000 tokens.

## Query Path for Agents

When 10 pre-loaded turns aren't enough, the agent can run:

```bash
# See project state, work items, process log
./myswat status -p tisql

# See raw recent turns (up to 50 persisted)
./myswat history -p tisql --turns 50

# Search knowledge base
./myswat memory search "error handling pattern" -p tisql
```

These commands read from TiDB and print to stdout, which the agent can consume in its
tool output.

## Files to Modify

| File | Change |
|------|--------|
| `db/migrations/v007_conversation_persistence.py` | New migration: add `compacted_at` column + turn recency index |
| `db/schema.py` | Register v006 (already exists as file) and v007 in `MIGRATION_MODULES` |
| `memory/store.py` | Add `get_recent_turns_by_project()` (window function query), `gc_compacted_turns()` (turn-only deletion, 0-based offset, `(created_at, id)` tie-breaker), `mark_session_fully_compacted()`. Remove `purge_compacted_sessions()` and `delete_archived_session()`. |
| `memory/retriever.py` | Replace fallback history with always-on Tier 1, add pre-load hint, remove `KNOWLEDGE_SUFFICIENCY_THRESHOLD`. Update `_build_current_session_context()` to ignore watermark (load all physically-present turns). No Tier 2 dedup needed (overlap accepted). |
| `memory/compactor.py` | `threshold_turns=50`, remove `threshold_tokens` parameter entirely. Replace all calls to `mark_session_compacted()` with `mark_session_fully_compacted()` (both in `compact_session(..., mark_compacted=True)` and in the short-session branch of `compact_all_pending()`). This ensures `compacted_at` is set, making sessions visible to `myswat gc`. **Fix failure ambiguity**: `compact_session()` currently treats both "no useful knowledge" and "failed AI parse" as `[]` and marks the session compacted either way. Change `parse_compaction_output()` to return `(items, ok)` tuple: `([], True)` for a valid empty JSON array (legitimate empty — AI found nothing to extract), `([], False)` for any parse failure (no JSON array found, malformed response). Then in `compact_session()`: call `mark_session_fully_compacted()` only when the AI call succeeded (`response.success`) AND parsing succeeded (`ok is True`). If the AI call failed or parsing failed (`ok is False`), leave the session as `completed` so it remains retryable by `compact_all_pending()`. **Callers to update for `(items, ok)` return type**: `memory/ingester.py:142` also calls `parse_compaction_output()` — update to unpack the tuple (ingester can ignore `ok` and just use `items`). **Tests to update**: `tests/test_memory/test_compactor.py` and `tests/test_memory/test_compactor_full.py` assert plain list returns — update to assert `(list, bool)` tuples. |
| `config/settings.py` | Remove `compaction.threshold_tokens` from `MySwatSettings` |
| `agents/session_manager.py` | `close()`: stop deleting sessions/turns. `_check_mid_session_compaction()`: remove `delete_compacted_turns()` call. Add `_compact_final()`. |
| `cli/chat_cmd.py` | Remove `threshold_tokens=` from `KnowledgeCompactor(...)` construction |
| `cli/run_cmd.py` | Remove `threshold_tokens=` from `KnowledgeCompactor(...)` construction |
| `cli/work_cmd.py` | Remove `threshold_tokens=` from `KnowledgeCompactor(...)` construction |
| `cli/main.py` | Add `myswat history` command, add `myswat gc` command |
| `cli/memory_cmd.py` | Remove `threshold_tokens=` from `KnowledgeCompactor(...)`. Remove `purge` command (replaced by `myswat gc`). |
