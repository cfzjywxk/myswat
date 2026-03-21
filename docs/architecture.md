# Architecture

## Components

| Component | Role |
|-----------|------|
| **CLI** (Typer) | Thin client for `init`, `work`, `status`, `chat`, `search`, etc. |
| **MySwatDaemon** | Persistent local daemon: project init, work submission, worker supervision |
| **WorkflowKernel** | Stage-oriented orchestration: queue tasks, wait for results, advance stages |
| **MySwatToolService** | Store-backed MCP tool surface for coordination |
| **Managed Worker** | `myswat worker --role ...` process running a Codex/Claude/Kimi backend |
| **SessionManager** | Chat/session lifecycle for transcript-oriented flows |
| **MemoryRetriever** | 5-tier context builder for agent prompts |
| **KnowledgeSearchEngine** | Lexical + vector + graph fusion via RRF |
| **MemoryStore** | TiDB CRUD, source-aware upsert, lexical index |

## Workflow

```
User submits requirement
  -> Daemon queues work item
  -> Daemon starts managed workers per role
  -> Workers claim stage assignments via MCP
  -> Agents execute (write code, run tests, review)
  -> WorkflowKernel advances or loops review stages
  -> Final report + persisted knowledge
```

MySwat routes prompts and captures coordination state. It does **not** run builds or tests — agents do that themselves.

## Memory & Knowledge

### Context Tiers (loaded at session start)

| Tier | What | Budget |
|------|------|--------|
| 0 | Project ops (build, test, conventions) + work item state | Always loaded |
| 1 | Recent cross-role turns (10 per role) | 25% |
| 2 | Knowledge search (lexical + vector + graph) | 50% |
| 3 | Current session continuity | 25% |
| 4 | Active work items + recent artifacts | 5% |

### Knowledge Write Paths

- **Session compaction** — AI extracts structured knowledge from conversation turns
- **Workflow learn triggers** — session/workflow summaries submit extraction jobs
- **Document ingestion** — code-boundary-aware chunking, optionally AI-distilled
- **Project ops seeding** — `myswat init` stores core workflow knowledge

### Knowledge Search

Three retrieval branches fused via weighted reciprocal-rank fusion (RRF):

- **Lexical** — inverted index (CamelCase, snake_case, `::` paths, `/` paths)
- **Vector** — BGE-M3 cosine similarity
- **Graph** — entity/relation expansion

Search modes: `auto`, `exact`, `concept`, `relation`. Profiles: `quick`, `standard`, `precise`.

### Conversation Persistence

```
Session active
  +-- All turns saved (always)
  +-- At 50 uncompacted turns -> mid-session compaction
  |     AI extracts knowledge, advances watermark
  |     Raw turns stay visible for pre-load
  +-- Session close
  |     Final compaction, status = 'compacted'
  |     Raw turns NOT deleted
  +-- GC (myswat gc)
        Deletes turns past grace period from compacted sessions
        Keeps most recent 50 turns per project
```

## TiDB Schema

| Table | Purpose |
|-------|---------|
| `projects` | Project registry |
| `agents` | Role configs per project |
| `work_items` | Task tracking |
| `stage_runs` | Workflow stage execution records |
| `coordination_events` | Append-only handoff/status/review stream |
| `runtime_registrations` | Runtime capability and heartbeat |
| `artifacts` | Proposals/diffs under review |
| `review_cycles` | Review iterations with verdicts |
| `sessions` | Dialog sessions with compaction watermark |
| `session_turns` | Individual messages |
| `knowledge` | Compacted/ingested knowledge with embeddings |
| `knowledge_terms` | Lexical term index |
| `knowledge_entities` | Entity anchors for graph expansion |
| `knowledge_relations` | Relationship graph |
| `document_sources` | Source-file tracking for incremental ingestion |
