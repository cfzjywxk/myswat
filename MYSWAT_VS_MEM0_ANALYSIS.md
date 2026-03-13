# MySwat vs Mem0: Memory Architecture Comparison

## Architecture At a Glance

| Dimension | **myswat** | **mem0** |
|-----------|-----------|---------|
| **Core model** | Conversation orchestrator — stores raw turns, batch-compacts to knowledge | Memory middleware — distills facts at write-time, discards raw turns |
| **What persists** | Raw turns (until GC) + compacted knowledge entries | Only extracted memory facts |
| **Compaction** | Batch at 50 turns + session close; AI extracts categorized knowledge | Per-`add()` call; LLM classifies ADD/UPDATE/DELETE/NOOP against existing memories |
| **Context injection** | 4-tier deterministic: project ops → recent turns → knowledge search → current session | Pull-based: `search(query)` → inject into system prompt |
| **Vector search** | BGE-M3 local (1024d) + TiDB VEC_COSINE_DISTANCE | OpenAI text-embedding-3-small (1536d) + 20+ vector DB backends |
| **Graph support** | None | Neo4j/Memgraph/Neptune — entity-relationship triples |
| **Scoping** | Project-scoped, cross-role (dev sees QA history) | user_id / agent_id / session scoping |
| **Code awareness** | Built for dev/QA agent pipelines, understands work items and review cycles | Generic — no code/repo/diff awareness |

---

## Use Case: Training a TiDB/TiKV Expert with Continuous Knowledge Feeding

### 1. Document & Source Code Ingestion

**Mem0 is weak here.** It's conversation-centric — `memory.add()` takes chat message pairs and extracts atomic facts. There's no repo indexer, no AST-aware chunking, no code-structure understanding. You'd have to manually format source code as "conversations" and feed it through, losing structural context. It works well for remembering "user prefers X" or "Alice works at Google" — not for indexing 18 invariants across a segmented Makefile.

**Myswat's `myswat learn` + ingester** is closer but still limited. It does document chunking + AI distillation into knowledge entries with categories (`architecture`, `pattern`, `decision`, etc.). The project_ops tier ensures build/test conventions are always injected deterministically. But the ingester isn't designed for bulk source-code indexing either — it's better at capturing project guidelines and operational knowledge.

**Verdict: Neither is great for bulk source code ingestion.** Both need augmentation. But myswat's categorized knowledge model (with `architecture`, `pattern`, `bug_fix` categories) maps better to what a TiDB/TiKV expert needs to retain.

### 2. Continuous Knowledge Accumulation

**Mem0's write-time consolidation** is elegant for incremental facts: each new piece of information is immediately compared against existing memories, and the LLM decides to merge/update/replace. This prevents duplication and handles contradictions automatically. Good for evolving knowledge like "TiKV uses raft-rs v0.7" → "TiKV migrated to raft-rs v0.8".

**Myswat's batch compaction** is coarser — 50 turns accumulate, then an AI pass extracts knowledge items. The watermark/GC model means raw turns stay queryable until explicitly cleaned up. This preserves richer context (exact wording, sequence of reasoning) at the cost of storage and retrieval complexity.

**Verdict: Mem0 handles incremental fact updates more gracefully.** Myswat preserves richer context history but doesn't auto-deduplicate knowledge.

### 3. Cross-Session Expert Knowledge Recall

**Myswat's 4-tier retrieval is more practical for structured dev workflows:**
- **Tier 0** (deterministic): TiDB build conventions, invariants, Makefile structure — always present, never lost in vector search noise
- **Tier 1** (recent turns): Last 10 turns per role — dev sees what QA flagged, QA sees what dev implemented
- **Tier 2** (knowledge search): Vector similarity on accumulated expertise
- **Tier 3** (current session): Full session context

This means your TiDB expert agent always gets the project's operational knowledge (deterministic), sees recent cross-role context (dev/QA awareness), and can search deeper knowledge on demand.

**Mem0's retrieval** is purely semantic search — you search, you get the most similar memories. No guarantee that critical operational knowledge surfaces unless the search query happens to match. No cross-role awareness built in.

**Verdict: Myswat is significantly better for expert recall in a structured dev workflow.** The deterministic tier ensures critical knowledge is never missed.

### 4. Graph Relationships (Mem0's Advantage)

Mem0's graph memory can capture relationships like:
- `TiKV` --uses--> `raft-rs` --version--> `0.7`
- `RocksDB` --is_storage_engine_of--> `TiKV`
- `Coprocessor` --handles--> `TiDB_pushdown_queries`

This structural knowledge is valuable for a TiDB/TiKV expert — understanding how components relate, dependency chains, architectural decisions. **Myswat has no equivalent.** Knowledge entries are flat text with tags and categories.

**Verdict: Mem0's graph layer is genuinely useful for domain expertise.** This is the one clear capability gap in myswat.

---

## How MySwat's Memory Layer Actually Works Today

### The Data Flow

```
User prompt → SessionManager.send()
                 ├─ First turn only: MemoryRetriever.build_context_for_agent()
                 │    ├─ Tier 0: myswat.md file (deterministic, always)
                 │    ├─ Tier 0b: Work item state (stage, TODOs, process log)
                 │    ├─ Tier 1: 10 recent turns per role, project-wide (SQL window function)
                 │    ├─ Tier 2: search_knowledge() — vector + keyword hybrid
                 │    ├─ Tier 3: Current session turns
                 │    └─ Tier 4: Active work items + recent artifacts
                 ├─ append_turn(user) → session_turns table
                 ├─ runner.invoke(prompt)
                 ├─ append_turn(assistant) → session_turns table
                 └─ _check_mid_session_compaction() at 50 turns
                      └─ KnowledgeCompactor → AI extracts JSON → knowledge table
```

### What Each Component Actually Does

**MemoryStore** (`memory/store.py`, ~900 LOC) — Pure CRUD. Writes turns, reads turns, stores knowledge entries with embeddings, runs hybrid search (VEC_COSINE_DISTANCE + LIKE). Has `gc_compacted_turns()` for cleanup. No intelligence — just SQL.

**MemoryRetriever** (`memory/retriever.py`, ~450 LOC) — Read-only context builder. Called once per AI session start. Assembles a markdown string from 5 sources with token budgets (25%/50%/25%/5%). Truncates aggressively (500 chars per turn in history, 800 in current session). No caching, no dedup.

**KnowledgeCompactor** (`memory/compactor.py`, ~237 LOC) — Batch extraction. Takes 50+ uncompacted turns, sends transcript to AI, parses JSON array of knowledge items, stores them, advances watermark. Categories: `decision`, `architecture`, `pattern`, `bug_fix`, `review_feedback`, `progress`, `lesson_learned`.

**DocumentIngester** (`memory/ingester.py`, ~195 LOC) — File chunking (8KB chunks with 500-char overlap) + AI distillation. Falls back to storing raw chunks if no AI runner. This is the "knowledge feeding" path. Categories: `architecture`, `pattern`, `decision`, `api_reference`, `configuration`, `lesson_learned`.

**SessionManager** (`agents/session_manager.py`, ~280 LOC) — Lifecycle glue. Create/resume sessions, delegate to retriever on first turn, save turns, trigger compaction, close with final compaction. Key insight: context is built ONCE per AI session, not per turn — the AI CLI maintains its own conversation state.

---

## The Three Critical Weaknesses (What to Learn from Mem0)

### 1. Knowledge is append-only — no dedup, no update, no conflict resolution

This is the biggest gap. `store_knowledge()` is a blind INSERT every time. When the compactor extracts "TiKV uses raft-rs for consensus" from session 1 and extracts "TiKV uses raft-rs v0.8 for consensus" from session 15, you get **two rows** in the knowledge table. After months of continuous feeding on TiDB/TiKV docs, you'll have hundreds of overlapping entries competing for the same Tier 2 search slots.

**Mem0's approach**: Every `add()` call retrieves the top-10 semantically similar existing memories, then an LLM classifies each extracted fact as ADD/UPDATE/DELETE/NOOP. Existing memories get merged with complementary info. Contradictions get resolved.

**What myswat should do**: Before inserting a new knowledge entry, search existing knowledge for semantic overlap (the vector search already exists). If similarity > threshold, ask the LLM to merge the old + new content into one updated entry. This belongs in `store_knowledge()` or as a wrapper in the compactor.

### 2. Document ingestion is naive — no structural awareness, no incremental update

`DocumentIngester.ingest_file()` does fixed-size text chunking. Feed it a 20K-line TiKV source file and it splits at paragraph boundaries, sends each 8KB chunk to an AI, and stores whatever it extracts. Problems:

- **No dedup across re-ingestion**: Feed the same doc twice, get double the entries. There's no `source_file` check before inserting.
- **No incremental update**: When the doc changes, there's no diff — you'd have to `delete_knowledge_by_category()` and re-ingest everything.
- **No structural awareness**: Rust source gets chunked the same as a README. No AST parsing, no function boundaries, no module awareness.
- **8KB chunks with 500-char overlap**: Arbitrary. A TiKV raft module definition might span 15KB and get split across chunks, losing context.

**Mem0's approach** doesn't solve the code-awareness problem either (it has no AST support), but its write-time dedup means re-feeding a document naturally updates existing memories instead of duplicating them.

**What myswat should do**:
- Track `source_file` + content hash. On re-ingest, only process changed chunks.
- For code files: chunk by logical units (functions, impl blocks) not byte offsets. Even a simple regex-based splitter (split on `fn `, `impl `, `mod `) would be better.
- Add an `upsert_knowledge()` method that searches for existing entries from the same source file and updates them instead of creating new rows.

### 3. No relationship/graph model — entities are disconnected facts

The knowledge table stores flat text entries with tags. There's no way to represent "RocksDB is-storage-engine-of TiKV" or "Coprocessor handles TiDB pushdown queries" as structured relationships. When the retriever searches, it can only find entries whose text happens to match the query vector — it can't traverse connections.

For a TiDB/TiKV expert, relationships are critical:
- `TiKV → uses → raft-rs → for → consensus`
- `TiDB → pushes down → expressions → to → Coprocessor`
- `Region → splits at → 96MB → configured by → region-max-size`

**Mem0's approach**: Graph memory with Neo4j/Memgraph — entity extraction, relationship triplets, temporal versioning of relationships.

**What myswat should do**: A `knowledge_relations` table in TiDB would work (no need for Neo4j):

```sql
CREATE TABLE knowledge_relations (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    project_id BIGINT NOT NULL,
    source_entity VARCHAR(255),
    relation VARCHAR(128),
    target_entity VARCHAR(255),
    source_knowledge_id BIGINT,
    confidence FLOAT DEFAULT 1.0,
    valid_from TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    valid_until TIMESTAMP NULL,  -- NULL = still valid
    INDEX idx_entity (project_id, source_entity),
    INDEX idx_target (project_id, target_entity)
);
```

Extract entity triples during compaction (add it to the `COMPACTION_PROMPT`) and during document ingestion. At retrieval time, when the task mentions "coprocessor", follow edges to find related components.

---

## Practical Recommendation

**For training a TiDB/TiKV expert with continuous feeding, myswat's architecture is more practical**, with key gaps to fill:

| What works today | What's missing |
|-----------------|---------------|
| Deterministic project ops injection (Tier 0) — build conventions always present | No bulk source-code/document ingestion pipeline |
| Cross-role context (Tier 1) — expert sees full dev/QA history | No graph/relationship model for component architecture |
| Categorized knowledge (Tier 2) — `architecture`, `pattern`, `bug_fix` map to domain expertise | No write-time deduplication (knowledge can overlap) |
| Agent-driven inspection — agents run `git diff`, read source directly | Ingester not optimized for large Rust/Go codebases |

### What to steal from mem0:

1. **Write-time dedup/update** — When compacting, compare new knowledge against existing entries and UPDATE/merge instead of blindly appending. Prevents knowledge store from growing stale duplicates over months of continuous feeding.

2. **Graph layer for architecture** — Add a lightweight entity-relationship model for TiDB/TiKV component topology. Even a `knowledge_relations` table in TiDB with `(source_entity, relation, target_entity, embedding)` would capture structural knowledge.

3. **Incremental fact extraction** — Instead of only extracting knowledge at 50-turn compaction boundaries, also extract key facts from individual important turns (e.g., when an agent discovers a new invariant or architectural pattern).

### What mem0 can't do that myswat needs:

- Orchestrate dev↔QA review loops with persistent cross-role context
- Deterministically inject project conventions (mem0 is all-search, no guarantees)
- Scope knowledge to work items and review cycles
- Let agents query deeper history via CLI (`myswat history`, `myswat status`)

---

## Priority Order for Implementation

| Change | Effort | Impact for TiDB/TiKV Expert |
|--------|--------|----------------------------|
| **Write-time dedup/merge** in knowledge store | Medium — add similarity search before insert, LLM merge prompt | **High** — prevents knowledge rot over months of continuous feeding |
| **Incremental document re-ingestion** — hash tracking, upsert | Low — add content_hash column, check before insert | **High** — lets you re-feed updated docs without duplication |
| **Code-aware chunking** — split at function/module boundaries | Medium — language-specific regex splitters | **Medium** — preserves semantic units in Rust/Go source |
| **Entity-relationship graph** | Medium — new table + extraction prompt additions | **Medium** — structural knowledge for component relationships |
| **Relevance decay + TTL** — already exists (`decay_relevance()`, `expire_stale_knowledge()`) | Already implemented | Just needs to be wired into a cron/schedule |

The first two are the most impactful and aren't hard. The knowledge store is already 90% there — it just needs the "compare before insert" step that mem0 does by default.

---

*Analysis date: 2026-03-13*
