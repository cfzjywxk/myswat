# Architecture

## Components

| Component | Role |
|-----------|------|
| **CLI** (Typer) | `myswat init`, `learn`, `chat`, `run`, `work`, `gc` |
| **SessionManager** | Agent lifecycle, subprocess launch (codex/claude/kimi) |
| **WorkflowEngine** | Mode dispatch: full, design, development, test |
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
- **Document ingestion** — files chunked (Rust/Go boundary-aware), optionally AI-distilled
- **Manual add** — `myswat memory add`
- **Project learn** — `myswat learn` (delete-and-replace, not upsert)

All paths except `learn` go through the **upsert pipeline**: content hash dedup, title-based merge candidate lookup, textual containment merge, optional LLM merge, supersede, or create.

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
