# Architecture

## Components

| Component | Role |
|-----------|------|
| **CLI** (Typer) | Thin clients for `init`, `work`, `status`, plus legacy `chat` / `run` paths |
| **MySwatDaemon** | Persistent local daemon that handles project init, work submission, and worker supervision |
| **WorkflowKernel** | Stage-oriented orchestration for `work`: queue tasks, wait for results, advance stages |
| **WorkflowRuntime** | Lightweight role/profile wrapper used by the kernel |
| **MySwatToolService** | Canonical store-backed coordination surface for workflow and MCP tools |
| **MySwat MCP Endpoint** | HTTP JSON-RPC MCP surface for runtime registration, task claim, artifact submission, and review verdicts |
| **Managed Worker** | Internal `myswat worker --role ...` process that executes Codex/Claude/Kimi backends for one role |
| **SessionManager** | Legacy chat/session lifecycle for transcript-oriented flows |
| **MemoryRetriever** | 5-tier context builder for agent prompts |
| **KnowledgeSearchEngine** | Search planning, lexical/vector/graph fusion |
| **MemoryStore** | TiDB CRUD, source-aware upsert, lexical index, graph metadata |
| **KnowledgeCompactor** | Distill session turns into reusable knowledge entries |
| **DocumentIngester** | Chunk and distill documents into knowledge (code-boundary-aware) |
| **Embedder** | BGE-M3 local embeddings for vector search (optional) |
| **GC** | Lazy garbage collection with 7-day grace period |

## Memory & Knowledge Layer

### Context Tiers (loaded at session start)

| Tier | What | Budget |
|------|------|--------|
| 0 | Project ops (build, test, conventions) | Always loaded |
| 0b | Current work item state | Always loaded |
| 1 | Recent cross-role turns (10 per role) | 25% |
| 2 | Knowledge search (lexical + vector + graph) | 50% |
| 3 | Current session continuity | 25% |
| 4 | Active work items + recent artifacts | 5% |

### Knowledge Write Paths

- **Session compaction** — AI extracts structured knowledge from conversation turns
- **Workflow and chat learn triggers** — chat/session/workflow summaries can submit learn jobs
- **Document ingestion** — files chunked (Rust/Go boundary-aware), optionally AI-distilled through the ingestion pipeline
- **Project ops seeding** — `myswat init` stores core team workflow knowledge for new projects

Knowledge created from sessions, workflow learns, and ingestion flows goes through the **upsert pipeline**: content hash dedup, title-based merge candidate lookup, textual containment merge, optional LLM merge, supersede, or create. Project-ops seeding uses its own title-aware replace/update path.

### Knowledge Search

Three retrieval branches fused via weighted reciprocal-rank fusion (RRF):

- **Lexical** — inverted index with structured tokenization (CamelCase, snake_case, `::` paths, `/` paths)
- **Vector** — BGE-M3 cosine similarity
- **Graph** — entity/relation expansion for related concepts

Search modes: `auto`, `exact`, `concept`, `relation`. Profiles: `quick`, `standard`, `precise`.

### Conversation Persistence Lifecycle

```
Session active
  |
  +-- All turns saved to session_turns (always)
  |
  +-- At 50 uncompacted turns --> mid-session compaction
  |     +-- AI extracts knowledge, advance watermark
  |     +-- Raw turns stay (visible for pre-load)
  |
  +-- Session close
  |     +-- status = 'completed', run final compaction
  |     +-- If fully compacted: status = 'compacted', set compacted_at
  |     +-- Raw turns NOT deleted
  |
  +-- GC (myswat gc, separate pass)
        +-- Deletes turns from compacted sessions past grace period
        +-- Keeps most recent 50 turns per project regardless
```

## TiDB Schema

| Table | Purpose |
|-------|---------|
| `projects` | Project registry with memory revision counter |
| `agents` | Role configs per project |
| `stage_runs` | Explicit workflow stage execution records |
| `coordination_events` | Append-only handoff/status/review event stream |
| `runtime_registrations` | Runtime capability and heartbeat records |
| `sessions` | Dialog sessions with compaction watermark and context revision |
| `session_turns` | Individual messages (recency-indexed) |
| `knowledge` | Compacted/ingested knowledge with embeddings, source metadata, hashes, merge lineage |
| `knowledge_terms` | Lexical term index for exact technical retrieval |
| `knowledge_entities` | Extracted entity anchors for graph expansion |
| `knowledge_relations` | Lightweight relationship graph between entities |
| `document_sources` | Source-file content tracking for incremental re-ingestion |
| `work_items` | Task tracking |
| `artifacts` | Proposals/diffs under review |
| `review_cycles` | Review iterations with structured verdicts |

## Workflow State

The primary state model for `myswat work` is now:

- `work_items`
- `stage_runs`
- `coordination_events`
- `artifacts`
- `review_cycles`

That means the `work` path no longer depends on transcript persistence or CLI session resume to coordinate multi-agent delivery. The daemon starts or reuses managed workers, the kernel queues stage and review assignments, and workers claim them through the MCP endpoint.
