# Memory Layer Improvement Design (v16)

> Goal: Make myswat's memory layer competitive with mem0 + AI agent solutions
> for the use case of **continuously training a TiDB/TiKV domain expert** that
> can later handle complex issues autonomously.
>
> v2: Addresses review findings on provenance safety, embedding consistency,
> migration atomicity, compaction ordering, category scoping, and relation
> conflict resolution.
>
> v3: Addresses cross-file document merge, graph retraction on source
> deletion, missing unique constraint in v010, and version CAS on merge.
>
> v4: Addresses multi-source relation provenance, v009/v010 deployment
> ordering, upsert_relation concurrency, and manual producer source_type.
>
> v5: Addresses migration registration in schema.py, reactivation/conflict
> ordering in upsert_relation, narrowed exception handling in
> gc_unsupported_relations_if_available, and manual retraction clarity.
>
> v6: Addresses cross-type concurrent race in upsert_relation via
> transaction + SELECT FOR UPDATE, and retraction table consistency.
>
> v7: Addresses GC/upsert race in gc_unsupported_relations via atomic
> UPDATE ... WHERE NOT EXISTS, and clarifies manual entry deletion path.
>
> v8: Removes standalone _add_relation_source() — all relation_sources
> writes go through upsert_relation's transaction to maintain the
> locking guarantee against gc_unsupported_relations().
>
> v9: gc_unsupported_relations() now participates in entity-row locking
> protocol, closing the in-flight transaction visibility gap.
> delete_knowledge_by_id path calls GC for consistency.
>
> v10: GC step 3 scoped to locked candidate IDs only — no longer
> project-wide, so unlocked entities can't be affected mid-flight.
>
> v11: No backward compatibility requirement. Single v008 migration
> (was v008/v009/v010). Removed gc_unsupported_relations_if_available(),
> backward-compatible parser, deployment-ordering workarounds, and
> _graph_enabled() conditionals.
>
> v12: Fixes "atomic" wording (migration is sequential, not transactional),
> parser call site updated to 3-tuple, stale gating text removed, entity
> processing added to compactor, findings history marked as historical.
>
> v13: Round 12 — GPT re-reviewed v11 (not v12); all 3 findings already
> fixed. Remaining gap: historical findings table had stale resolutions
> referencing three migrations, backward-compatible parser, gc_if_available,
> and per-phase schema.py. All superseded entries now struck through with
> cross-references to the round that replaced them.
>
> v14: Round 13 — entity triple provenance fix. Each triple now carries
> `knowledge_index` mapping to its supporting knowledge item instead of
> blindly linking all triples to the last-created entry.
>
> v15: Round 14 — index-preserving provenance. Replaced compressed
> `created_ids` list with `kid_by_index` dict so skipped items don't
> shift indices and break knowledge_index lookups.
>
> v16: Round 15 — skip triples with no valid provenance instead of
> creating unprovenanced relations that linger until an unrelated GC.

## Problem Statement

After months of continuous knowledge feeding (documents, source code, agent
conversations), the current myswat memory layer degrades:

1. **Knowledge table bloats with duplicates** — `store_knowledge()` is blind
   INSERT. The same fact extracted from session 1 and session 50 produces two
   rows. Vector search returns stale duplicates instead of the freshest,
   most complete version.

2. **Re-ingesting updated docs doubles entries** — `DocumentIngester` has no
   content tracking. Feed TiKV's `raftstore.rs` docs today and again after a
   refactor: you get 2x the entries, half outdated.

3. **No structural relationships** — Knowledge entries are flat text. "RocksDB
   is the storage engine of TiKV" and "Coprocessor pushes down expressions
   from TiDB" exist as disconnected paragraphs. The retriever can't follow
   edges from one concept to related concepts.

4. **Code chunking is file-unaware** — 8KB fixed chunks split Rust functions
   mid-body. A 200-line `fn apply_snapshot()` might land across two chunks
   with no overlap capturing the full signature + body.

## Design Principles

- **No new external dependencies** — everything runs in TiDB + local Python.
  No Neo4j, no Redis, no external graph DB. TiDB is the single backend.
- **No backward compatibility requirement** — clean-slate migration. All
  schema changes are in a single v008 migration (one version, not
  transactional — statements execute sequentially; version recorded only
  on full success). No dual-format parsers, no deployment-ordering
  workarounds, no try/except for missing tables.
- **LLM-assisted but LLM-optional** — dedup/merge uses LLM when available,
  falls back to deterministic heuristics (vector similarity threshold +
  content hash) when not. The system must work without AI for basic dedup.
- **Provenance-safe** — merge never crosses source-type boundaries. File
  knowledge and session knowledge are independently retractable.

---

## Core Model: Source-Type Isolation

> **Key invariant**: Knowledge entries from different source types are NEVER
> merged into the same row. This ensures any source can be cleanly retracted
> without corrupting knowledge from other sources.

### Source types

| `source_type` | Producer | Retraction method |
|---------------|----------|-------------------|
| `"session"` | `KnowledgeCompactor` | N/A (session knowledge accumulates, corrected by newer sessions) |
| `"document"` | `DocumentIngester` | `delete_knowledge_by_source_file()` on re-ingest |
| `"manual"` | `myswat learn`, `memory add` | `project_ops` category: bulk via `delete_knowledge_by_category()`; other categories: individual via `myswat memory delete <id>` (new command, Phase 1) |

### Merge scoping rules

1. **Same source_type** AND **same category** → eligible for merge
2. **Document-sourced entries additionally require same `source_file`** —
   knowledge from `raftstore.rs` never merges with knowledge from
   `coprocessor.rs`, even if both are `source_type='document'` and
   `category='architecture'`. This ensures file retraction is clean.
3. **Session-sourced entries** merge freely within the same category
   (sessions don't have a retractable unit — they accumulate and self-correct)
4. **Different source_type** → never merge (skip or create new entry)
5. **Different category** → never merge
6. **Transient categories** (`progress`, `review_feedback`) → never merge
   targets; they are write-once observations that age out via relevance decay

**Retraction unit** = the smallest provenance scope that can be cleanly
deleted:

| source_type | Retraction unit | Merge scope |
|-------------|----------------|-------------|
| `document` | `source_file` | same `source_type` + `category` + `source_file` |
| `session` | none (accumulates) | same `source_type` + `category` |
| `manual` | `project_ops`: bulk by category; other categories: individual via `memory delete <id>` | same `source_type` + `category` |

This solves the provenance problem fully: when `raftstore.rs` is updated, we
`DELETE FROM knowledge WHERE source_file = ? AND source_type = 'document'`.
This deletes only rows whose content came entirely from `raftstore.rs`.
Knowledge from `coprocessor.rs` or sessions is untouched — it was never
merged into `raftstore.rs`-owned rows.

---

## Improvement 1: Write-Time Knowledge Dedup/Merge

### What mem0 does

Every `memory.add()` retrieves top-10 similar existing memories, then an LLM
classifies each extracted fact as ADD / UPDATE / DELETE / NOOP. Existing
memories get merged with complementary info. Contradictions get resolved.

### What myswat does differently

Mem0 merges freely across all sources — it stores only distilled facts with
no retraction capability. Myswat must support retraction (file re-ingestion,
knowledge correction) so it **scopes merge to same source_type + category**.

### Schema changes (v008)

```sql
-- Classify the origin of each knowledge entry
ALTER TABLE knowledge ADD COLUMN source_type VARCHAR(32) DEFAULT 'session';
-- Values: 'session', 'document', 'manual'

-- Track content identity for fast exact-match dedup
ALTER TABLE knowledge ADD COLUMN content_hash CHAR(64) DEFAULT NULL;
CREATE INDEX idx_knowledge_content_hash ON knowledge (project_id, content_hash);

-- Track merge lineage (which entries were absorbed into this one)
ALTER TABLE knowledge ADD COLUMN merged_from JSON DEFAULT NULL;
-- e.g. [{"id": 42, "title": "...", "merged_at": "2026-03-14T..."}]

-- Track version for optimistic concurrency on merge
ALTER TABLE knowledge ADD COLUMN version INT DEFAULT 1;

-- Backfill: infer source_type from existing data
UPDATE knowledge SET source_type = 'document' WHERE source_file IS NOT NULL;
UPDATE knowledge SET source_type = 'manual' WHERE category = 'project_ops';
-- Remaining rows (source_session_id IS NOT NULL) stay as default 'session'
```

### New method: `MemoryStore.upsert_knowledge()`

```python
# Categories that should never be merge targets — they are write-once
# observations that age out naturally via relevance decay.
TRANSIENT_CATEGORIES = frozenset({"progress", "review_feedback"})

# Categories that represent stable expert knowledge — merge-eligible.
STABLE_CATEGORIES = frozenset({
    "decision", "architecture", "pattern", "bug_fix",
    "lesson_learned", "api_reference", "configuration",
})


def upsert_knowledge(
    self,
    project_id: int,
    category: str,
    title: str,
    content: str,
    *,
    source_type: str = "session",
    # Existing params forwarded to store_knowledge()
    agent_id: int | None = None,
    source_session_id: int | None = None,
    source_turn_ids: list[int] | None = None,
    source_file: str | None = None,
    tags: list[str] | None = None,
    relevance_score: float = 1.0,
    confidence: float = 1.0,
    ttl_days: int | None = None,
    compute_embedding: bool = True,
    # Dedup control
    similarity_threshold: float = 0.85,
    merge_runner: AgentRunner | None = None,
) -> tuple[int, str]:
    """Insert, merge, or skip a knowledge entry. Returns (knowledge_id, action).

    Action is one of: "created", "merged", "skipped".

    Merge constraints (provenance-safe):
    - Only merges within the SAME source_type (session↔session, document↔document)
    - Only merges within the SAME category
    - Document-sourced entries additionally require same source_file
      (raftstore.rs never merges with coprocessor.rs)
    - Transient categories (progress, review_feedback) are never merge targets
    - Embedding is ALWAYS recomputed after merge
    - Version CAS guard prevents concurrent merges from overwriting each other

    Algorithm:
    1. If category is transient → skip dedup, always INSERT.
    2. Compute content_hash (SHA-256 of normalized content).
    3. Exact-match: if content_hash exists in same merge scope,
       return ("skipped", existing_id).
    4. Semantic similarity: search existing entries within merge scope.
    5. If top match similarity > threshold:
       a. LLM merge (if runner available) or deterministic merge.
       b. UPDATE existing row with CAS guard (WHERE version = expected).
       c. RECOMPUTE embedding on the merged content.
       d. If CAS fails (concurrent merge), fall through to INSERT.
       e. Return ("merged", existing_id).
    6. Else: INSERT new row with embedding. Return ("created", new_id).

    Merge scope per source_type:
    - document: project + category + source_type + source_file
    - session:  project + category + source_type
    - manual:   project + category + source_type
    """
    import hashlib
    content_hash = hashlib.sha256(content.strip().encode()).hexdigest()

    # Determine merge scope — document entries are scoped to source_file
    merge_source_file = source_file if source_type == "document" else None

    # 1. Transient categories always insert without dedup
    if category in TRANSIENT_CATEGORIES:
        kid = self.store_knowledge(
            project_id=project_id, category=category, title=title,
            content=content, source_type=source_type,
            agent_id=agent_id, source_session_id=source_session_id,
            source_turn_ids=source_turn_ids, source_file=source_file,
            tags=tags, relevance_score=relevance_score,
            confidence=confidence, ttl_days=ttl_days,
            compute_embedding=compute_embedding, content_hash=content_hash,
        )
        return kid, "created"

    # 2. Exact-match dedup (within merge scope)
    existing = self.find_by_content_hash(
        project_id, category, source_type, content_hash,
        source_file=merge_source_file,
    )
    if existing:
        return existing["id"], "skipped"

    # 3. Semantic similarity search (within merge scope)
    candidates = self.find_similar_knowledge(
        project_id, category, source_type, title, content,
        source_file=merge_source_file, limit=5,
    )
    if candidates and candidates[0]["similarity"] > similarity_threshold:
        best = candidates[0]
        # 4. Merge
        merged = self._merge_knowledge(
            existing=best, new_title=title, new_content=content,
            new_tags=tags, new_relevance=relevance_score,
            merge_runner=merge_runner,
        )
        if merged["action"] == "keep":
            return best["id"], "skipped"

        # 5. UPDATE with CAS guard + recomputed embedding
        updated = self._update_knowledge_with_reembed(
            knowledge_id=best["id"],
            expected_version=best.get("version", 1),
            title=merged["title"],
            content=merged["content"],
            tags=merged["tags"],
            relevance_score=merged["relevance_score"],
            merged_from_entry=best,
            content_hash=hashlib.sha256(merged["content"].strip().encode()).hexdigest(),
        )
        if updated:
            return best["id"], "merged"
        # CAS failed (concurrent merge) — fall through to INSERT
        # This is safe: worst case we get a near-duplicate that will
        # merge on the next compaction/ingestion pass.

    # 6. No match (or CAS miss) → create new entry
    kid = self.store_knowledge(
        project_id=project_id, category=category, title=title,
        content=content, source_type=source_type,
        agent_id=agent_id, source_session_id=source_session_id,
        source_turn_ids=source_turn_ids, source_file=source_file,
        tags=tags, relevance_score=relevance_score,
        confidence=confidence, ttl_days=ttl_days,
        compute_embedding=compute_embedding, content_hash=content_hash,
    )
    return kid, "created"
```

### Embedding recomputation on merge

The current `store_knowledge()` computes embeddings only on INSERT. After
merge, the content changes but the old embedding remains — making the entry
harder to find via vector search. This is fixed by `_update_knowledge_with_reembed()`:

```python
def _update_knowledge_with_reembed(
    self,
    knowledge_id: int,
    expected_version: int,
    title: str,
    content: str,
    tags: list[str] | None,
    relevance_score: float,
    merged_from_entry: dict,
    content_hash: str,
) -> bool:
    """Update a knowledge entry's content AND recompute its embedding.

    Uses optimistic concurrency (CAS on version column):
    - UPDATE ... WHERE id = %s AND version = %s
    - If another merge incremented version first, this returns False
      and the caller falls through to INSERT instead of overwriting.

    Returns True if the row was updated, False if CAS failed.
    """
    import json

    # Recompute embedding for the merged content
    vec_sql, embed_args = embedder.resolve_embed_sql(
        f"{title}\n{content}", self._tidb_embedding_model,
    )

    # Build merged_from lineage
    existing_merged = merged_from_entry.get("merged_from")
    if isinstance(existing_merged, str):
        existing_merged = json.loads(existing_merged)
    if not isinstance(existing_merged, list):
        existing_merged = []
    existing_merged.append({
        "id": merged_from_entry["id"],
        "title": merged_from_entry.get("title", "")[:100],
        "merged_at": datetime.now().isoformat(timespec="seconds"),
    })

    sql = (
        f"UPDATE knowledge SET title = %s, content = %s, "
        f"embedding = {vec_sql}, "
        f"tags = %s, relevance_score = %s, content_hash = %s, "
        f"merged_from = %s, version = version + 1 "
        f"WHERE id = %s AND version = %s"
    )
    args = [
        title, content,
        *embed_args,
        json.dumps(tags) if tags else None,
        relevance_score,
        content_hash,
        json.dumps(existing_merged[-10:]),  # cap lineage
        knowledge_id,
        expected_version,
    ]
    rows_affected = self._pool.execute(sql, tuple(args))
    return rows_affected > 0
```

### Scoped similarity search

```python
def find_similar_knowledge(
    self,
    project_id: int,
    category: str,
    source_type: str,
    title: str,
    content: str,
    source_file: str | None = None,
    limit: int = 5,
) -> list[dict]:
    """Find semantically similar entries within the merge scope.

    For document-sourced entries, source_file narrows the scope so that
    knowledge from raftstore.rs is never considered a merge candidate for
    coprocessor.rs. For session-sourced entries, source_file is None and
    merge scope is the full project+category+source_type.
    """
    vec_sql, vec_args = embedder.resolve_embed_sql(
        f"{title}\n{content}", self._tidb_embedding_model,
    )
    if vec_sql == "NULL":
        return self._find_similar_keyword(
            project_id, category, source_type, title, content,
            source_file, limit,
        )

    where = (
        "k.project_id = %s AND k.category = %s AND k.source_type = %s "
        "AND k.embedding IS NOT NULL "
        "AND (k.expires_at IS NULL OR k.expires_at > NOW())"
    )
    where_args: list = [project_id, category, source_type]

    if source_file is not None:
        where += " AND k.source_file = %s"
        where_args.append(source_file)

    sql = (
        f"SELECT k.*, "
        f"1.0 - VEC_COSINE_DISTANCE(k.embedding, {vec_sql}) AS similarity "
        f"FROM knowledge k "
        f"WHERE {where} "
        f"ORDER BY similarity DESC LIMIT %s"
    )
    args = [*vec_args, *where_args, limit]
    return self._pool.fetch_all(sql, tuple(args))


def find_by_content_hash(
    self,
    project_id: int,
    category: str,
    source_type: str,
    content_hash: str,
    source_file: str | None = None,
) -> dict | None:
    """Find an exact content match within the merge scope."""
    sql = (
        "SELECT * FROM knowledge WHERE project_id = %s AND category = %s "
        "AND source_type = %s AND content_hash = %s"
    )
    args: list = [project_id, category, source_type, content_hash]

    if source_file is not None:
        sql += " AND source_file = %s"
        args.append(source_file)

    sql += " LIMIT 1"
    return self._pool.fetch_one(sql, tuple(args))
```

### Merge logic

```python
def _merge_knowledge(
    self,
    existing: dict,
    new_title: str,
    new_content: str,
    new_tags: list[str] | None,
    new_relevance: float,
    merge_runner: AgentRunner | None = None,
) -> dict:
    """Merge two entries. Returns dict with action, title, content, tags, relevance_score.

    Uses LLM when available, deterministic heuristics otherwise.
    """
    if merge_runner:
        return self._llm_merge(existing, new_title, new_content, new_tags, merge_runner)
    return self._deterministic_merge(existing, new_title, new_content, new_tags, new_relevance)


@staticmethod
def _deterministic_merge(existing, new_title, new_content, new_tags, new_relevance):
    """Merge without LLM — conservative strategy."""
    old_content = existing.get("content", "")

    if len(new_content) > len(old_content) * 1.3:
        # New is substantially longer — likely more complete, replace
        merged_content = new_content
        merged_title = new_title
    elif len(old_content) > len(new_content) * 1.3:
        # Old is substantially longer — keep it
        return {"action": "keep", "title": "", "content": "", "tags": [], "relevance_score": 0}
    else:
        # Similar length — append update section
        merged_content = (
            old_content.rstrip()
            + "\n\n---\n[Updated]\n"
            + new_content
        )
        merged_title = new_title  # prefer newer title

    existing_tags = set(existing.get("tags") or [])
    new_tag_set = set(new_tags or [])
    merged_tags = list(existing_tags | new_tag_set)[:8]

    return {
        "action": "merge",
        "title": merged_title,
        "content": merged_content,
        "tags": merged_tags,
        "relevance_score": max(
            existing.get("relevance_score", 0.5), new_relevance,
        ),
    }
```

### Merge prompt (when LLM is available)

```python
KNOWLEDGE_MERGE_PROMPT = """You are a knowledge curator. Given two knowledge entries
about the same topic, merge them into a single comprehensive entry.

EXISTING ENTRY:
Title: {existing_title}
Content: {existing_content}
Tags: {existing_tags}

NEW ENTRY:
Title: {new_title}
Content: {new_content}
Tags: {new_tags}

Output a JSON object with:
- "action": one of "merge" (combine both), "replace" (new supersedes old),
  "keep" (existing is already complete, discard new)
- "title": merged title (max 100 chars)
- "content": merged content (1-3 paragraphs, keep all non-redundant detail)
- "tags": merged tag list (max 8)
- "relevance_score": float 0.0-1.0

Output ONLY the JSON object. No other text.
"""
```

### Integration: fix compaction ordering (Finding 4)

Current `compactor.py` advances watermark BEFORE storing knowledge (line 178
before line 192). If `upsert_knowledge()` fails, turns are marked compacted
but knowledge is lost.

**Fix**: store all knowledge entries first, advance watermark only after all
succeed.

```python
# compactor.py — compact_session() reordered:

items, entity_triples, ok = parse_compaction_output(response.content)
if not ok:
    return []

last_included_turn_index = included_turns[-1].turn_index
last_uncompacted_turn_index = new_turns[-1].turn_index
turn_ids = [t.id for t in included_turns if t.id is not None]

# Store knowledge entries FIRST (before advancing watermark).
# Index-preserving map: kid_by_index[i] = knowledge ID for items[i],
# or None if items[i] was skipped (empty content).
kid_by_index: dict[int, int] = {}
for i, item in enumerate(items):
    category = item.get("category", "progress")
    title = item.get("title", "Untitled")[:512]
    content = item.get("content", "")
    if not content:
        continue
    tags = item.get("tags", [])
    relevance = min(max(float(item.get("relevance_score", 0.8)), 0.0), 1.0)
    confidence = min(max(float(item.get("confidence", 0.8)), 0.0), 1.0)

    kid, action = self._store.upsert_knowledge(
        project_id=project_id,
        category=category,
        title=title,
        content=content,
        source_type="session",
        agent_id=None,
        source_session_id=session_id,
        source_turn_ids=turn_ids,
        tags=tags,
        relevance_score=relevance,
        confidence=confidence,
        merge_runner=self._runner,
    )
    kid_by_index[i] = kid

created_ids = list(kid_by_index.values())

# Store entity-relationship triples extracted from the same transcript.
# Each triple carries a knowledge_index (0-based into the original items
# array) mapping it to the knowledge item it was derived from.
# Triples with missing/invalid knowledge_index are skipped — never create
# active relations without provenance.
for triple in entity_triples:
    ki = triple.get("knowledge_index")
    source_kid = kid_by_index.get(ki) if ki is not None else None
    if source_kid is None:
        continue  # skip: no valid provenance → don't pollute the graph
    src_id = self._store.upsert_entity(
        project_id, triple["source"], triple["source_type"],
    )
    tgt_id = self._store.upsert_entity(
        project_id, triple["target"], triple["target_type"],
    )
    self._store.upsert_relation(
        project_id, src_id, triple["relation"], tgt_id,
        source_knowledge_id=source_kid,
    )

# Advance watermark ONLY AFTER all knowledge entries are persisted
# If storing failed above, watermark stays put → turns will be
# re-compacted next time.
self._store.advance_compaction_watermark(session_id, last_included_turn_index)

if mark_compacted and last_included_turn_index == last_uncompacted_turn_index:
    self._store.mark_session_fully_compacted(session_id)
return created_ids
```

### Expected behavior over time

Session 1 compaction → "TiKV uses raft-rs for consensus" →
  **created** (id=1, source_type=session, category=architecture)

Session 5 compaction → "TiKV uses raft-rs v0.8" →
  similarity 0.91 to id=1, same source_type+category →
  LLM merge → "TiKV uses raft-rs v0.8 for consensus, migrated from v0.7" →
  **merged** into id=1, embedding recomputed

Session 20 compaction → "raft-rs consensus in TiKV" →
  similarity 0.93 → LLM says "keep" → **skipped**

File ingestion → `raft-rs/README.md` → "raft-rs provides Raft consensus" →
  similarity 0.89 to id=1 BUT different source_type (document vs session) →
  **created** as new entry (id=50, source_type=document)

Later re-ingest of `raft-rs/README.md` →
  `delete_knowledge_by_source_file(raft-rs/README.md)` deletes id=50 only →
  id=1 (session-derived) untouched

---

## Improvement 2: Incremental Document Ingestion

### Problem

`DocumentIngester.ingest_file()` has no memory of what it previously ingested.
Feeding the same file twice doubles entries. Feeding an updated file creates
new entries alongside stale old ones.

### Schema changes

```sql
CREATE TABLE IF NOT EXISTS ingested_documents (
    id           BIGINT AUTO_INCREMENT PRIMARY KEY,
    project_id   BIGINT NOT NULL,
    source_path  VARCHAR(1024) NOT NULL,
    content_hash CHAR(64) NOT NULL,       -- SHA-256 of full file content
    file_size    BIGINT DEFAULT 0,
    chunk_count  INT DEFAULT 0,
    knowledge_count INT DEFAULT 0,        -- entries created from this file
    language     VARCHAR(32) DEFAULT NULL, -- "rust", "go", "markdown", etc.
    ingested_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_project_source (project_id, source_path),
    INDEX idx_content_hash (project_id, content_hash),
    FOREIGN KEY (project_id) REFERENCES projects(id)
);
```

### Retraction-safe ingestion flow

File re-ingestion deletes ONLY document-sourced entries from that file.
Session-derived entries that happen to cover the same topic are untouched
because they have `source_type = 'session'`.

```python
def ingest_file(self, file_path: str, project_id: int, ...) -> IngestResult:
    """Ingest a file with change detection and safe retraction.

    Returns IngestResult with action taken and knowledge IDs.
    """
    content = Path(file_path).read_text()
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    # 1. Check if already ingested with same content
    existing = self._store.get_ingested_document(project_id, file_path)
    if existing and existing["content_hash"] == content_hash:
        return IngestResult(action="unchanged", knowledge_ids=[])

    # 2. If previously ingested with different content → retract old entries
    #    SAFE: only deletes rows where source_type='document' AND source_file=path
    #    CASCADE on relation_sources removes source links, then GC invalidates
    #    relations that lost all sources.
    if existing:
        self._store.delete_knowledge_by_source_file(project_id, file_path)
        self._store.gc_unsupported_relations(project_id)

    # 3. Detect language and chunk accordingly
    language = detect_language(file_path)
    if language in ("rust", "go"):
        chunks = chunk_code(content, language)
    else:
        chunks = chunk_text(content)

    # 4. Ingest chunks with source_type='document'
    #    upsert_knowledge will merge within document-sourced entries only
    knowledge_ids = []
    for i, chunk in enumerate(chunks):
        ids = self._ingest_chunk(
            chunk, i+1, len(chunks),
            source_type="document",  # scopes all merge to document-only
            ...
        )
        knowledge_ids.extend(ids)

    # 5. Record ingestion state
    self._store.upsert_ingested_document(
        project_id, file_path, content_hash,
        len(content), len(chunks), len(knowledge_ids), language,
    )
    return IngestResult(
        action="updated" if existing else "created",
        knowledge_ids=knowledge_ids,
    )
```

### Entity extraction in ingester

`_ingest_chunk` mirrors the compactor's entity handling: the AI prompt includes
`ENTITY_EXTRACTION_SUFFIX`, and the response is parsed with
`parse_compaction_output()`. The same `knowledge_index` + skip-if-unprovenanced
pattern applies:

```python
def _ingest_chunk(self, chunk, chunk_index, total_chunks,
                  project_id, source_type, source_file, ...):
    """Ingest a single chunk: extract knowledge + entity triples."""
    prompt = self._build_chunk_prompt(chunk, chunk_index, total_chunks)
    response = self._runner.invoke(prompt)
    items, entity_triples, ok = parse_compaction_output(response.content)
    if not ok:
        return []

    # Store knowledge entries, preserving index for triple provenance
    kid_by_index: dict[int, int] = {}
    for i, item in enumerate(items):
        content = item.get("content", "")
        if not content:
            continue
        kid, _ = self._store.upsert_knowledge(
            project_id=project_id,
            source_type=source_type,
            source_file=source_file,
            ...  # category, title, tags from item
        )
        kid_by_index[i] = kid

    # Store entity triples — skip if no valid provenance
    for triple in entity_triples:
        ki = triple.get("knowledge_index")
        source_kid = kid_by_index.get(ki) if ki is not None else None
        if source_kid is None:
            continue
        src_id = self._store.upsert_entity(
            project_id, triple["source"], triple["source_type"],
        )
        tgt_id = self._store.upsert_entity(
            project_id, triple["target"], triple["target_type"],
        )
        self._store.upsert_relation(
            project_id, src_id, triple["relation"], tgt_id,
            source_knowledge_id=source_kid,
        )

    return list(kid_by_index.values())
```

### Store methods for retraction

```python
def delete_knowledge_by_source_file(self, project_id: int, source_file: str) -> int:
    """Delete knowledge entries from a specific source file.

    SAFE: only deletes document-sourced entries. Session-derived entries
    that cover the same topic remain untouched.

    Graph retraction is handled automatically by the ON DELETE CASCADE on
    relation_sources.knowledge_id FK: when knowledge rows are deleted,
    their relation_sources rows are cascade-deleted. This removes the
    knowledge entry as a supporting source for any relations it backed.
    Afterwards, gc_unsupported_relations() should be called to invalidate
    relations that have lost all their sources.

    """
    return self._pool.execute(
        "DELETE FROM knowledge WHERE project_id = %s "
        "AND source_file = %s AND source_type = 'document'",
        (project_id, source_file),
    )


def gc_orphaned_entities(self, project_id: int, dry_run: bool = False) -> int:
    """Delete entities with no active relations pointing to/from them.

    Entities are shared across sources (e.g., "TiKV" is referenced by
    both document and session knowledge). They should only be deleted
    when no active relation references them, not when a single source
    is retracted.

    Call from `myswat gc` alongside turn GC.
    """
    orphans = self._pool.fetch_all(
        "SELECT ke.id FROM knowledge_entities ke "
        "WHERE ke.project_id = %s "
        "AND NOT EXISTS ("
        "  SELECT 1 FROM knowledge_relations kr "
        "  WHERE (kr.source_entity_id = ke.id OR kr.target_entity_id = ke.id) "
        "  AND kr.valid_until IS NULL"
        ")",
        (project_id,),
    )
    if dry_run or not orphans:
        return len(orphans)

    orphan_ids = [r["id"] for r in orphans]
    placeholders = ", ".join(["%s"] * len(orphan_ids))
    # Delete invalidated (historical) relations referencing orphaned entities first
    self._pool.execute(
        f"DELETE FROM knowledge_relations "
        f"WHERE source_entity_id IN ({placeholders}) "
        f"OR target_entity_id IN ({placeholders})",
        (*orphan_ids, *orphan_ids),
    )
    self._pool.execute(
        f"DELETE FROM knowledge_entities WHERE id IN ({placeholders})",
        tuple(orphan_ids),
    )
    return len(orphan_ids)
```

### CLI: extend existing `myswat feed` command

The `myswat feed` command already exists in `cli/feed_cmd.py`. This design
**extends** it (not introduces it) with change detection and status reporting.

```bash
# Existing behavior preserved — but now with change detection
myswat feed -p tikv path/to/raftstore.rs

# New: directory feed with progress (already supported, now incremental)
myswat feed -p tikv src/server/ --glob "*.rs"
# Output:
#   [1/45] raftstore.rs [rust] → unchanged, skipping
#   [2/45] coprocessor.rs [rust] → updated, 4 chunks, 3 entries
#   ...

# New: force re-ingestion
myswat feed -p tikv src/server/ --force

# New: show ingestion status
myswat feed -p tikv --status
```

---

## Improvement 3: Code-Aware Chunking

### Problem

Fixed 8KB character chunks split code at arbitrary byte offsets. A Rust
function definition might span 15KB and get split across two chunks with no
semantic boundary awareness.

### Language-aware splitters

New file `memory/chunkers.py` — regex-based splitting at logical code
boundaries. No full AST parsing required.

```python
# memory/chunkers.py

import re
from dataclasses import dataclass
from pathlib import Path

@dataclass
class CodeChunk:
    content: str
    chunk_type: str   # "function", "impl_block", "module", "struct", "text"
    name: str | None  # e.g. "fn apply_snapshot" or "impl RaftStore"
    line_start: int
    line_end: int


def detect_language(file_path: str) -> str | None:
    ext_map = {
        ".rs": "rust", ".go": "go", ".py": "python",
        ".md": "markdown", ".toml": "toml", ".yaml": "yaml",
    }
    return ext_map.get(Path(file_path).suffix.lower())


def chunk_code(content: str, language: str, max_size: int = 12000) -> list[CodeChunk]:
    if language == "rust":
        return _chunk_rust(content, max_size)
    elif language == "go":
        return _chunk_go(content, max_size)
    else:
        return _chunk_text_as_code(content, max_size)


def _chunk_rust(content: str, max_size: int) -> list[CodeChunk]:
    """Split Rust source at fn/impl/mod/struct boundaries.

    Strategy:
    1. Split at top-level item boundaries (pub fn, impl, mod, struct, enum, trait).
    2. If an item exceeds max_size, split at inner fn boundaries.
    3. Always include the preceding doc comment (/// or //!) with the item.
    """
    item_pattern = re.compile(
        r'^(?:(?:\s*///.*\n)*)'           # doc comments
        r'(?:\s*#\[.*\]\n)*'             # attributes
        r'\s*(?:pub(?:\(crate\))?\s+)?'  # visibility
        r'(?:async\s+)?'                 # async
        r'(?:fn|impl|mod|struct|enum|trait|type|const|static)\s',
        re.MULTILINE,
    )

    boundaries = [m.start() for m in item_pattern.finditer(content)]
    if not boundaries:
        return _chunk_text_as_code(content, max_size)

    if boundaries[0] > 0:
        boundaries.insert(0, 0)

    chunks = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(content)
        segment = content[start:end].rstrip()
        if not segment.strip():
            continue

        line_start = content[:start].count('\n') + 1
        line_end = line_start + segment.count('\n')
        name = _extract_rust_item_name(segment)
        chunk_type = _classify_rust_item(segment)

        if len(segment) <= max_size:
            chunks.append(CodeChunk(segment, chunk_type, name, line_start, line_end))
        else:
            sub_chunks = _split_oversized_rust(segment, max_size, line_start)
            chunks.extend(sub_chunks)

    return chunks if chunks else _chunk_text_as_code(content, max_size)


def _chunk_go(content: str, max_size: int) -> list[CodeChunk]:
    """Split Go source at func/type/var/const boundaries."""
    item_pattern = re.compile(
        r'^(?:(?:\s*//.*\n)*)'
        r'\s*(?:func|type|var|const)\s',
        re.MULTILINE,
    )
    boundaries = [m.start() for m in item_pattern.finditer(content)]
    if not boundaries:
        return _chunk_text_as_code(content, max_size)

    if boundaries[0] > 0:
        boundaries.insert(0, 0)

    chunks = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(content)
        segment = content[start:end].rstrip()
        if not segment.strip():
            continue
        line_start = content[:start].count('\n') + 1
        line_end = line_start + segment.count('\n')
        name = _extract_go_item_name(segment)

        if len(segment) <= max_size:
            chunks.append(CodeChunk(segment, "function", name, line_start, line_end))
        else:
            for j, sub in enumerate(_chunk_text_as_code(segment, max_size)):
                chunks.append(CodeChunk(
                    sub.content, "function_part",
                    f"{name} (part {j+1})" if name else None,
                    line_start, line_end,
                ))

    return chunks if chunks else _chunk_text_as_code(content, max_size)
```

### Integration with DocumentIngester

```python
# In _ingest_chunk(), use CodeChunk metadata for better titles:
if isinstance(chunk, CodeChunk):
    title = f"{source_name}::{chunk.name}" if chunk.name else f"{source_name} L{chunk.line_start}"
else:
    title = f"{source_name} (chunk {chunk_index})"
```

### What this means for TiDB/TiKV

Feeding `src/server/raftstore/store.rs` (~5000 lines) produces:
```
store.rs::impl RaftStore           [impl_block]  L45-L280
store.rs::fn apply_snapshot        [function]    L282-L410
store.rs::fn handle_raft_ready     [function]    L412-L650
store.rs::fn on_region_heartbeat   [function]    L652-L780
```

Each chunk is a complete logical unit. Vector search for "how does TiKV apply
snapshots" directly matches `fn apply_snapshot` instead of hoping it lands in
the right 8KB window.

---

## Improvement 4: Entity-Relationship Graph

### Problem

Flat knowledge entries can't represent structural relationships. When the
retriever gets "coprocessor", it finds entries mentioning "coprocessor" but
can't traverse to "TiDB expression pushdown" or "RocksDB scan iterators".

### Schema changes

```sql
CREATE TABLE IF NOT EXISTS knowledge_entities (
    id           BIGINT AUTO_INCREMENT PRIMARY KEY,
    project_id   BIGINT NOT NULL,
    name         VARCHAR(255) NOT NULL,
    entity_type  VARCHAR(64) NOT NULL,
    -- "component", "concept", "api", "config", "file", "module"
    description  TEXT,
    embedding    VECTOR,
    created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_project_name_type (project_id, name, entity_type),
    INDEX idx_project_type (project_id, entity_type),
    FOREIGN KEY (project_id) REFERENCES projects(id)
);

CREATE TABLE IF NOT EXISTS knowledge_relations (
    id                    BIGINT AUTO_INCREMENT PRIMARY KEY,
    project_id            BIGINT NOT NULL,
    source_entity_id      BIGINT NOT NULL,
    relation              VARCHAR(128) NOT NULL,
    target_entity_id      BIGINT NOT NULL,
    confidence            FLOAT DEFAULT 1.0,
    valid_from            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    valid_until           TIMESTAMP NULL,   -- NULL = still valid
    created_at            TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_relation_tuple (project_id, source_entity_id, relation, target_entity_id),
    INDEX idx_source (project_id, source_entity_id),
    INDEX idx_target (project_id, target_entity_id),
    INDEX idx_relation (project_id, relation),
    FOREIGN KEY (project_id) REFERENCES projects(id),
    FOREIGN KEY (source_entity_id) REFERENCES knowledge_entities(id),
    FOREIGN KEY (target_entity_id) REFERENCES knowledge_entities(id)
);

-- Junction table: which knowledge entries support each relation.
-- A relation is valid as long as at least one supporting source exists.
-- Retraction removes one source; if sources become empty the relation
-- is invalidated.
CREATE TABLE IF NOT EXISTS relation_sources (
    relation_id     BIGINT NOT NULL,
    knowledge_id    BIGINT NOT NULL,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (relation_id, knowledge_id),
    FOREIGN KEY (relation_id) REFERENCES knowledge_relations(id) ON DELETE CASCADE,
    FOREIGN KEY (knowledge_id) REFERENCES knowledge(id) ON DELETE CASCADE
);
```

**Graph retraction model**: Relations have **multi-source provenance** via
the `relation_sources` junction table. Multiple knowledge entries (from
different files, sessions, etc.) can independently support the same relation
tuple. Retraction removes one source; the relation stays active as long as
at least one source remains.

```
                    knowledge (id=10, doc A)  ──┐
                                                ├──► relation_sources ──► relation "TiKV uses raft-rs"
                    knowledge (id=25, session)  ──┘

Delete doc A → CASCADE deletes relation_sources row for id=10
            → relation still has source from id=25 → stays active
            → gc_unsupported_relations() is a no-op for this relation

Delete doc A AND session compacted away →
            → relation_sources is empty for this relation
            → gc_unsupported_relations() invalidates it (valid_until = NOW())
```

Lifecycle:
```
Knowledge created → entities upserted → relation upserted →
  relation_sources row added (relation_id, knowledge_id)

File re-ingested →
  1. Old knowledge rows deleted → CASCADE removes their relation_sources rows
  2. gc_unsupported_relations() invalidates relations with zero remaining sources
  3. New knowledge created → new relations created / old reactivated
  4. gc_orphaned_entities() cleans entities with no active relations
```

### Automatic relation conflict resolution

When a relation is upserted between an entity pair, any existing active
relation **of a different type** between the same pair is automatically
invalidated **first** — before checking for an existing same-type tuple.
This ensures that reactivating an invalidated relation still invalidates
conflicting types. Same-type relations add a source via `relation_sources`.

**Concurrency**: The entire operation runs within a transaction with
`SELECT FOR UPDATE` on the source entity row. This serializes concurrent
writers operating on the same entity pair, preventing the race where two
writers both pass invalidation and both INSERT different relation types.
The `uk_relation_tuple` unique index only prevents same-tuple duplicates —
it cannot enforce pair-level type exclusivity, so application-level
serialization is required.

```python
def upsert_relation(
    self,
    project_id: int,
    source_entity_id: int,
    relation: str,
    target_entity_id: int,
    source_knowledge_id: int | None = None,
    confidence: float = 1.0,
) -> int:
    """Insert or update a relation with multi-source provenance.

    Rules:
    1. Invalidate conflicting active relations (same entity pair, different
       relation type) — this MUST happen first, before reactivation
    2. Same (source, relation, target) exists (active or invalidated) →
       reactivate if needed, add source to relation_sources, update confidence
    3. New tuple → insert + add source

    Concurrency: runs within a transaction with SELECT FOR UPDATE on the
    source entity row. This serializes writers operating on the same entity
    pair, preventing cross-type races. The uk_relation_tuple unique index
    prevents same-tuple duplicates as a safety net.
    """
    with self._pool.transaction() as tx:
        # 0. Serialize concurrent writers for the same entity pair.
        #    SELECT FOR UPDATE acquires a row lock on the source entity;
        #    a concurrent writer for the same pair (same source_entity_id)
        #    blocks here until this transaction commits. This prevents:
        #    - Cross-type race: two writers both pass invalidation and both
        #      INSERT different relation types, leaving two active types
        #    - Same-type race: two writers both INSERT the same tuple
        #      (also caught by uk_relation_tuple as a safety net)
        tx.fetch_one(
            "SELECT id FROM knowledge_entities WHERE id = %s FOR UPDATE",
            (source_entity_id,),
        )

        # 1. Invalidate conflicting active relations (same entity pair, different type).
        tx.execute(
            "UPDATE knowledge_relations SET valid_until = NOW() "
            "WHERE project_id = %s AND source_entity_id = %s "
            "AND target_entity_id = %s AND valid_until IS NULL "
            "AND relation != %s",
            (project_id, source_entity_id, target_entity_id, relation),
        )

        # 2. Check for existing tuple (active or invalidated)
        existing = tx.fetch_one(
            "SELECT id, valid_until FROM knowledge_relations "
            "WHERE project_id = %s AND source_entity_id = %s "
            "AND relation = %s AND target_entity_id = %s",
            (project_id, source_entity_id, relation, target_entity_id),
        )
        if existing:
            rid = existing["id"]
            if existing["valid_until"] is not None:
                tx.execute(
                    "UPDATE knowledge_relations SET valid_until = NULL, "
                    "valid_from = NOW(), confidence = %s WHERE id = %s",
                    (confidence, rid),
                )
            else:
                tx.execute(
                    "UPDATE knowledge_relations SET confidence = GREATEST(confidence, %s) "
                    "WHERE id = %s",
                    (confidence, rid),
                )
            if source_knowledge_id is not None:
                tx.execute(
                    "INSERT IGNORE INTO relation_sources (relation_id, knowledge_id) "
                    "VALUES (%s, %s)",
                    (rid, source_knowledge_id),
                )
            return rid

        # 3. Insert new relation
        rid = tx.insert_returning_id(
            "INSERT INTO knowledge_relations (project_id, source_entity_id, relation, "
            "target_entity_id, confidence) "
            "VALUES (%s, %s, %s, %s, %s)",
            (project_id, source_entity_id, relation, target_entity_id, confidence),
        )

        # 4. Add supporting source
        if source_knowledge_id is not None:
            tx.execute(
                "INSERT IGNORE INTO relation_sources (relation_id, knowledge_id) "
                "VALUES (%s, %s)",
                (rid, source_knowledge_id),
            )
        return rid


```

> **No standalone `_add_relation_source()`**: All relation-source writes go
> through `upsert_relation()`, which runs within a transaction with
> `SELECT FOR UPDATE` on the source entity. This ensures every
> `relation_sources` INSERT participates in the same locking protocol,
> so `gc_unsupported_relations()` can never invalidate a relation that
> a concurrent writer is about to support. If a future need arises for
> batch re-linking outside `upsert_relation()`, it must use the same
> `_pool.transaction()` + entity-row lock pattern.

### TiDBPool.transaction() — required for upsert_relation

`upsert_relation()` requires all steps to run on a single connection with
`autocommit=False` for `SELECT FOR UPDATE` locking. The existing `TiDBPool`
uses `autocommit=True` with per-statement connections. A `transaction()`
context manager is needed:

```python
# In db/connection.py — TiDBPool

@contextmanager
def transaction(self) -> Generator[TiDBTransaction, None, None]:
    """Execute multiple statements in a single transaction.

    Yields a TiDBTransaction with the same execute/fetch_one/fetch_all/
    insert_returning_id interface as TiDBPool. All operations run on
    one connection with autocommit=False. Commits on clean exit,
    rolls back on exception.
    """
    conn = self._connect()
    conn.autocommit(False)
    try:
        yield TiDBTransaction(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


class TiDBTransaction:
    """Transaction-scoped wrapper with the same query API as TiDBPool."""

    def __init__(self, conn: pymysql.Connection) -> None:
        self._conn = conn

    def execute(self, sql: str, args: tuple | None = None) -> int:
        with self._conn.cursor() as cur:
            cur.execute(sql, args)
            return cur.rowcount

    def fetch_one(self, sql: str, args: tuple | None = None) -> dict | None:
        with self._conn.cursor() as cur:
            cur.execute(sql, args)
            return cur.fetchone()

    def fetch_all(self, sql: str, args: tuple | None = None) -> list[dict]:
        with self._conn.cursor() as cur:
            cur.execute(sql, args)
            return cur.fetchall()

    def insert_returning_id(self, sql: str, args: tuple | None = None) -> int:
        with self._conn.cursor() as cur:
            cur.execute(sql, args)
            cur.execute("SELECT LAST_INSERT_ID() AS id")
            return cur.fetchone()["id"]
```

```python
def gc_unsupported_relations(self, project_id: int, dry_run: bool = False) -> int:
    """Invalidate active relations that have zero remaining sources.

    Called after knowledge deletion (e.g., file re-ingestion) to clean up
    relations whose last supporting knowledge entry was removed.

    Does NOT delete — sets valid_until = NOW() for temporal history.

    Concurrency: runs within a transaction and acquires FOR UPDATE locks
    on the source entity rows of candidate relations — the same locks
    that upsert_relation() acquires. This ensures:
    1. Any in-flight upsert_relation() on the same entities has either
       committed (its relation_sources rows visible) or is blocked.
    2. The final NOT EXISTS UPDATE sees latest committed data (DML reads
       current data, not the snapshot, under InnoDB/TiDB pessimistic mode).
    3. No relation is invalidated while a concurrent writer is mid-flight.

    Entity locks are acquired in sorted ID order to prevent deadlocks
    (upsert_relation() locks only one entity, so no circular wait).
    """
    if dry_run:
        row = self._pool.fetch_one(
            "SELECT COUNT(*) AS cnt FROM knowledge_relations kr "
            "WHERE kr.project_id = %s AND kr.valid_until IS NULL "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM relation_sources rs WHERE rs.relation_id = kr.id"
            ")",
            (project_id,),
        )
        return row["cnt"] if row else 0

    with self._pool.transaction() as tx:
        # 1. Find candidate unsupported relations and their source entities.
        candidates = tx.fetch_all(
            "SELECT kr.id, kr.source_entity_id "
            "FROM knowledge_relations kr "
            "WHERE kr.project_id = %s AND kr.valid_until IS NULL "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM relation_sources rs WHERE rs.relation_id = kr.id"
            ")",
            (project_id,),
        )
        if not candidates:
            return 0

        # 2. Acquire entity-row locks in sorted order.
        #    This serializes with any in-flight upsert_relation() on
        #    the same entities. After acquiring all locks, any concurrent
        #    upsert that was inserting relation_sources has either
        #    committed (rows now visible to DML) or is blocked on us.
        entity_ids = sorted(set(c["source_entity_id"] for c in candidates))
        for eid in entity_ids:
            tx.fetch_one(
                "SELECT id FROM knowledge_entities WHERE id = %s FOR UPDATE",
                (eid,),
            )

        # 3. Re-check NOT EXISTS only for the locked candidate IDs.
        #    Scoped to candidate IDs (not project-wide) so we only
        #    invalidate relations whose source entities we actually
        #    locked. Relations outside this set are untouched — even if
        #    they became unsupported during this window, their entities
        #    were not locked and a concurrent upsert could be mid-flight.
        candidate_ids = [c["id"] for c in candidates]
        placeholders = ", ".join(["%s"] * len(candidate_ids))
        return tx.execute(
            f"UPDATE knowledge_relations SET valid_until = NOW() "
            f"WHERE id IN ({placeholders}) AND valid_until IS NULL "
            f"AND NOT EXISTS ("
            f"  SELECT 1 FROM relation_sources rs "
            f"  WHERE rs.relation_id = knowledge_relations.id"
            f")",
            tuple(candidate_ids),
        )
```

### Compaction output format

The compaction prompt requires output as a JSON object with knowledge items
and entity triples. No backward compatibility with the old JSON array format.

```python
def parse_compaction_output(raw_output: str) -> tuple[list[dict], list[dict], bool]:
    """Parse AI output into knowledge items and entity triples.

    Returns (knowledge_items, entity_triples, ok).
    Expected format: {"knowledge": [...], "entities": [...]}
    """
    text = raw_output.strip()

    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            part = part.strip()
            if part.startswith("{"):
                text = part
                break

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return [], [], False
        else:
            return [], [], False

    if isinstance(parsed, dict):
        knowledge = parsed.get("knowledge", [])
        entities = parsed.get("entities", [])
        if isinstance(knowledge, list):
            if not isinstance(entities, list):
                entities = []
            return knowledge, entities, True

    return [], [], False
```

### Entity extraction prompt suffix

Always appended to `COMPACTION_PROMPT` and `INGESTION_PROMPT`. The suffix
is additive — it does not replace the existing prompt:

```python
ENTITY_EXTRACTION_SUFFIX = """

Additionally, extract entity-relationship triples from the content.

For each triple, output a JSON object with:
- "source": entity name (e.g. "TiKV", "RocksDB", "Coprocessor")
- "source_type": one of "component", "concept", "api", "config", "file", "module"
- "relation": verb (e.g. "uses", "contains", "depends_on", "configured_by", "handles")
- "target": entity name
- "target_type": same as source_type
- "knowledge_index": 0-based index into the "knowledge" array identifying which
  knowledge item this triple is derived from. If a triple spans multiple items,
  use the most specific one.

Wrap your output in a JSON object:
{"knowledge": [... items as before ...], "entities": [... triples ...]}

If no entities, use: {"knowledge": [...], "entities": []}
"""
```

The suffix is always appended to the compaction and ingestion prompts.
The parser expects the new `{"knowledge": [...], "entities": [...]}` format.

---

## Improvement 5: Graph-Enhanced Retrieval

### Enhanced retrieval flow

Add a graph expansion step between Tier 1 and Tier 2 in the retriever:

```python
def build_context_for_agent(self, project_id, agent_id, task_description, ...):
    # ... Tier 0, Tier 1 unchanged ...

    # ── 1.5: Entity graph expansion ──
    expanded_terms = self._expand_via_graph(project_id, task_description)

    # ── 2. Knowledge search (augmented with graph expansion) ──
    if task_description:
        results = self._store.search_knowledge(project_id, task_description, ...)
        # Graph-expanded secondary searches
        if expanded_terms and len(results) < 10:
            seen_ids = {r["id"] for r in results}
            for term in expanded_terms[:3]:
                extra = self._store.search_knowledge(project_id, term, limit=3)
                for r in extra:
                    if r["id"] not in seen_ids:
                        results.append(r)
                        seen_ids.add(r["id"])

    # ... rest unchanged ...


def _expand_via_graph(self, project_id: int, task_description: str) -> list[str]:
    """Extract entity mentions from task, expand via graph.

    No LLM call — pure string matching + SQL traversal.
    """
    if not task_description:
        return []

    entities = self._store.list_entities(project_id)
    if not entities:
        return []

    # Find known entities mentioned in the task
    task_lower = task_description.lower()
    mentioned = [e["name"] for e in entities if e["name"].lower() in task_lower]
    if not mentioned:
        return []

    # Expand each via 1-hop graph traversal
    expanded = []
    for name in mentioned[:5]:
        related = self._store.get_related_entities(project_id, name, max_depth=1, limit=5)
        for rel in related:
            expanded.append(f"{rel['related_entity']} ({rel['relation']} {name})")

    return expanded
```

### Example

Task: "fix the region split panic in TiKV"

1. Entity matching: `Region`, `TiKV`
2. Graph expansion for `Region`: `SplitChecker`, `RaftStore`, `region-max-size`
3. Tier 2 primary search: "region split panic" → finds 5 entries
4. Tier 2 expanded search: "SplitChecker (split_triggered_by Region)" → finds
   2 more entries about split checker logic that pure vector search missed

---

## Migration Plan: Single v008 Migration

No backward compatibility requirement — all schema changes in one migration
version. Not transactional: statements execute sequentially and the version
is recorded only after all succeed. If mid-migration failure occurs, manual
intervention is needed (see note below).

```python
VERSION = 8
DESCRIPTION = "Memory layer: dedup, ingestion tracking, entity graph"

STATEMENTS = [
    # ── Knowledge dedup columns ──
    "ALTER TABLE knowledge ADD COLUMN source_type VARCHAR(32) DEFAULT 'session'",
    "ALTER TABLE knowledge ADD COLUMN content_hash CHAR(64) DEFAULT NULL",
    "ALTER TABLE knowledge ADD COLUMN merged_from JSON DEFAULT NULL",
    "ALTER TABLE knowledge ADD COLUMN version INT DEFAULT 1",
    "CREATE INDEX idx_knowledge_content_hash ON knowledge (project_id, content_hash)",
    # Backfill source_type from existing data
    "UPDATE knowledge SET source_type = 'document' WHERE source_file IS NOT NULL",
    "UPDATE knowledge SET source_type = 'manual' WHERE category = 'project_ops'",

    # ── Ingested document tracking ──
    """
    CREATE TABLE IF NOT EXISTS ingested_documents (
        id           BIGINT AUTO_INCREMENT PRIMARY KEY,
        project_id   BIGINT NOT NULL,
        source_path  VARCHAR(1024) NOT NULL,
        content_hash CHAR(64) NOT NULL,
        file_size    BIGINT DEFAULT 0,
        chunk_count  INT DEFAULT 0,
        knowledge_count INT DEFAULT 0,
        language     VARCHAR(32) DEFAULT NULL,
        ingested_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uk_project_source (project_id, source_path),
        INDEX idx_content_hash (project_id, content_hash),
        FOREIGN KEY (project_id) REFERENCES projects(id)
    )
    """,

    # ── Entity graph ──
    """
    CREATE TABLE IF NOT EXISTS knowledge_entities (
        id           BIGINT AUTO_INCREMENT PRIMARY KEY,
        project_id   BIGINT NOT NULL,
        name         VARCHAR(255) NOT NULL,
        entity_type  VARCHAR(64) NOT NULL,
        description  TEXT,
        embedding    VECTOR,
        created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uk_project_name_type (project_id, name, entity_type),
        INDEX idx_project_type (project_id, entity_type),
        FOREIGN KEY (project_id) REFERENCES projects(id)
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS knowledge_relations (
        id                    BIGINT AUTO_INCREMENT PRIMARY KEY,
        project_id            BIGINT NOT NULL,
        source_entity_id      BIGINT NOT NULL,
        relation              VARCHAR(128) NOT NULL,
        target_entity_id      BIGINT NOT NULL,
        confidence            FLOAT DEFAULT 1.0,
        valid_from            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        valid_until           TIMESTAMP NULL,
        created_at            TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uk_relation_tuple (project_id, source_entity_id, relation, target_entity_id),
        INDEX idx_source (project_id, source_entity_id),
        INDEX idx_target (project_id, target_entity_id),
        INDEX idx_relation (project_id, relation),
        FOREIGN KEY (project_id) REFERENCES projects(id),
        FOREIGN KEY (source_entity_id) REFERENCES knowledge_entities(id),
        FOREIGN KEY (target_entity_id) REFERENCES knowledge_entities(id)
    )
    """,

    # Multi-source provenance: which knowledge entries support each relation.
    # ON DELETE CASCADE on knowledge_id means when a knowledge row is deleted
    # (e.g., file re-ingestion), its relation_sources rows are auto-removed.
    # The relation itself stays — gc_unsupported_relations() invalidates
    # relations that have lost all their sources.
    """
    CREATE TABLE IF NOT EXISTS relation_sources (
        relation_id     BIGINT NOT NULL,
        knowledge_id    BIGINT NOT NULL,
        created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (relation_id, knowledge_id),
        FOREIGN KEY (relation_id) REFERENCES knowledge_relations(id) ON DELETE CASCADE,
        FOREIGN KEY (knowledge_id) REFERENCES knowledge(id) ON DELETE CASCADE
    )
    """,
]
```

**Note**: ALTER TABLE statements are not individually idempotent in TiDB
(no `ADD COLUMN IF NOT EXISTS`). If v008 fails mid-way, manual intervention
is needed. The ALTER statements are ordered so the most likely-to-fail
(INDEX creation) comes last, and the UPDATE backfills and CREATE TABLE IF
NOT EXISTS statements are idempotent.

---

## Implementation Phases

All schema changes are in a single v008 migration. Phases are development
milestones, not separate deployments.

### Phase 1: Knowledge Dedup/Merge + Schema Migration

**Files changed:**
- `db/migrations/v008_memory_layer.py` — single migration (all tables + columns)
- `db/schema.py` — add `myswat.db.migrations.v008_memory_layer` to `MIGRATION_MODULES`
- `db/connection.py` — add `TiDBTransaction` class and `TiDBPool.transaction()` context manager
- `models/knowledge.py` — add `source_type`, `content_hash`, `merged_from`, `version`
- `memory/store.py` — add `upsert_knowledge()`, `find_similar_knowledge()`,
  `find_by_content_hash()`, `_update_knowledge_with_reembed()`, `_merge_knowledge()`,
  `delete_knowledge_by_id()`
- `memory/compactor.py` — reorder watermark advancement after knowledge storage,
  switch to `upsert_knowledge(source_type="session")`
- `memory/ingester.py` — switch to `upsert_knowledge(source_type="document")`
- `cli/learn_cmd.py` — add `source_type='manual'` to `store_knowledge()` calls (lines 307, 332)
- `cli/memory_cmd.py` — add `source_type='manual'` to `store_knowledge()` call (line 88);
  add `memory delete <id>` command (calls `delete_knowledge_by_id()` then
  `gc_unsupported_relations()`, same pattern as file retraction)

**Tests:**
- Exact-match dedup: same content_hash + same merge scope → skip
- Semantic dedup: similarity > 0.85 within same merge scope → merge
- Cross-source isolation: similarity > 0.85 but different source_type → create new
- Cross-file isolation: similarity > 0.85, same source_type=document, different
  source_file → create new (not merge)
- Same-file merge: two chunks from same file with high similarity → merge
- Category isolation: same content but different category → create new
- Transient category: `progress` entries never merge, always create
- Embedding recomputed: after merge, vector search finds the updated entry
- CAS guard: concurrent merge on same entry → second merge fails CAS, falls
  through to INSERT; next pass will merge the near-duplicate
- Compaction ordering: watermark not advanced until knowledge stored
- Merge lineage: merged_from JSON tracks absorbed entries
- TiDBPool.transaction(): commit on clean exit, rollback on exception,
  connection closed in both cases

**Definition of done:** After compacting 10 sessions discussing the same
TiKV raft topic, the knowledge table has 1 rich entry (not 10). Feeding a
doc about the same topic creates a second independent entry with
`source_type=document`. Feeding two different files about the same topic
creates two independent document entries (one per file).

### Phase 2: Incremental Ingestion + Code Chunking

**Files changed:**
- `memory/store.py` — add `get_ingested_document()`, `upsert_ingested_document()`,
  update `delete_knowledge_by_source_file()` to filter `source_type='document'`
- `memory/ingester.py` — rewrite `ingest_file()` with change detection
- `memory/chunkers.py` — **new file**, Rust/Go code-aware splitting
- `cli/feed_cmd.py` — extend with `--force`, `--status`, progress reporting

**Tests:**
- Feed file → creates entries + tracking record
- Feed same file again → "unchanged", no new entries
- Modify file, feed again → deletes old entries, creates new ones
- File retraction: session-derived entries untouched after file re-ingest
- Rust chunking: functions aren't split mid-body
- Go chunking: func boundaries respected
- Oversized items: gracefully split at inner boundaries

**Definition of done:** `myswat feed -p tikv src/` processes 500 Rust files.
Re-running is a no-op unless files changed. Updating one file retracts only
its document-sourced entries.

### Phase 3: Entity Graph + Enhanced Retrieval

**Files changed:**
- `memory/store.py` — add entity/relation CRUD with auto-conflict resolution,
  `gc_unsupported_relations()`, `gc_orphaned_entities()`
- `memory/compactor.py` — extend prompt with `ENTITY_EXTRACTION_SUFFIX`,
  update `parse_compaction_output()` for new object format
- `memory/ingester.py` — extract entities from document chunks
- `memory/retriever.py` — add `_expand_via_graph()`, augment Tier 2 search
- `cli/main.py` — add `myswat graph` command

**Tests:**
- Entity extraction from compaction: entities + relations stored
- Upsert idempotency: same entity twice → one row
- Auto-conflict: new relation between same pair with different type →
  old relation gets valid_until, new one active
- Same-type relation: update in place (confidence/source updated)
- Reactivation: invalidated relation with same tuple → reactivated
  (valid_until cleared) instead of duplicate INSERT
- Reactivation with conflict: (A, uses, B) invalidated, (A, depends_on, B)
  active → upsert_relation(A, uses, B) invalidates depends_on FIRST, then
  reactivates uses; both steps happen in one call
- Cross-type concurrent race: two threads upsert different relation types
  for the same entity pair → FOR UPDATE lock serializes them → only one
  active type at end (last-writer-wins)
- Same-tuple concurrent race: two threads upsert same (source, relation,
  target) → FOR UPDATE lock serializes → second writer finds existing
  row and adds source (no duplicate INSERT)
- Multi-source provenance: two knowledge entries support same relation →
  deleting one leaves relation active (one source remains)
- Multi-source provenance: delete both supporting knowledge entries →
  relation_sources empty → gc_unsupported_relations() invalidates the relation
- CASCADE retraction: DELETE FROM knowledge WHERE source_file = ? →
  relation_sources rows auto-removed, relation stays if other sources remain
- Graph retraction: delete all source knowledge for a relation → relation
  invalidated by GC, entities preserved (shared), orphaned entities cleaned
- File re-ingest with entities: old relations may lose one source via CASCADE,
  new relations created (may reactivate same tuples), entities never deleted
- Graph traversal: 1-hop and 2-hop expansion
- Retriever augmentation: graph-expanded search finds entries plain search misses
- gc_orphaned_entities: entities with no active relations → deleted in
  non-dry-run; historical relations cleaned first
- Concurrent upsert_relation: two writers race on same pair → FOR UPDATE
  serializes; both cross-type and same-type races produce correct results
- No standalone _add_relation_source: all relation_sources writes go
  through upsert_relation's transaction (no unprotected INSERT path)
- GC/upsert serialization: concurrent upsert_relation() and
  gc_unsupported_relations() on the same entity → entity-row lock
  serializes them; GC re-checks NOT EXISTS after lock acquisition,
  so relation is not invalidated if upsert committed in the interim
- memory delete <id>: calls delete_knowledge_by_id() then
  gc_unsupported_relations(), unsupported relations cleaned

**Definition of done:** After feeding TiKV docs + 20 dev sessions,
`myswat graph -p tikv --entity RaftStore` shows connected components.
Re-ingesting a changed file invalidates old relations and creates new ones
without losing session-derived graph edges. Retriever uses graph expansion.

### Phase 4: CLI Polish + Observability

**Files changed:**
- `cli/feed_cmd.py` — verbose progress, status reporting
- `cli/main.py` — `myswat graph`, `myswat knowledge stats`

**New commands:**
```bash
myswat knowledge stats -p tikv
#   Knowledge: 342 entries (47 merged, 12 skipped)
#     session-sourced: 180, document-sourced: 150, manual: 12
#     Stable: 298 (architecture: 89, pattern: 67, decision: 55, ...)
#     Transient: 44 (progress: 30, review_feedback: 14)
#   Entities: 89 | Relations: 156 (142 active, 14 superseded)
#   Ingested files: 23 (18 Rust, 3 Go, 2 Markdown)

myswat graph -p tikv --entity "RaftStore"
#   RaftStore [module]
#     ├── contains → apply_snapshot [api]
#     ├── contains → handle_raft_ready [api]
#     ├── managed_by ← Region [concept]
#     ├── uses → raft-rs [component]
#     └── depends_on → RocksDB [component]
```

---

## Review Findings Addressed (Historical)

> **Note**: This section records the design's evolution through review
> rounds. Earlier resolutions may have been superseded by later rounds —
> for example, "three migrations" (round 1) was collapsed to a single
> migration in round 10. The authoritative design is the main body above;
> this section exists for audit trail only.

### Round 1 (v2)

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | Cross-source merge corrupts provenance, makes file retraction unsafe | High | **Source-type isolation**: merge only within same `source_type` + `category`. File retraction deletes `source_type='document'` rows only. Session knowledge untouched. |
| 2 | Merge updates content but never recomputes embedding | High | **`_update_knowledge_with_reembed()`**: every merge path recomputes embedding via `embedder.resolve_embed_sql()` on the merged content. |
| 3 | Single v008 migration unsafe with current runner | High | ~~Three migrations (v008, v009, v010)~~ **Superseded by round 10**: single v008 migration, sequential execution, not transactional. Manual cleanup documented for partial-failure. |
| 4 | Compaction advances watermark before storing knowledge | High | **Reordered**: knowledge entries stored FIRST, watermark advanced AFTER all succeed. Failed store → watermark stays → turns re-compacted next time. |
| 5 | Merge scope too broad — transient categories merged into stable | Medium | **Category + stability scoping**: merge only within same category. `TRANSIENT_CATEGORIES` (`progress`, `review_feedback`) always INSERT, never merge targets. |
| 6 | Relation invalidation promised but not designed for auto-conflict | Medium | **Auto-conflict in `upsert_relation()`**: same entity pair with different relation type → old relation gets `valid_until=NOW()`, new one inserted. Same type → update in place. |
| OQ | Parser output format change not called out | — | ~~Backward-compatible parser~~ **Superseded by round 10**: parser accepts only `{"knowledge":[], "entities":[]}` format. No backward-compatible fallback. |
| OQ | "add myswat feed" but command already exists | — | **Corrected**: design extends existing `feed_cmd.py` with change detection, `--force`, `--status`. Not introducing a new command. |

### Round 2 (v3)

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | Cross-file document merge: two files can collapse into one row, retraction of one file corrupts the other | High | **Document merge scoped to same `source_file`**: `find_similar_knowledge()` and `find_by_content_hash()` accept `source_file` parameter. For `source_type='document'`, merge candidates must have the same `source_file`. `raftstore.rs` knowledge never merges with `coprocessor.rs` knowledge. Retraction unit table added to design. |
| 2 | Graph data not retracted when source knowledge deleted: FK has no ON DELETE, entities have no provenance, relations left dangling | High | **End-to-end graph retraction** (v3: `source_knowledge_id` FK with `ON DELETE SET NULL`; **v4: replaced with `relation_sources` junction table** — `ON DELETE CASCADE` on `relation_sources.knowledge_id` auto-removes source links, `gc_unsupported_relations()` invalidates relations with zero remaining sources). `gc_orphaned_entities()` cleans entities with no active relations. Entities are shared — never deleted on single-source retraction. |
| 3 | v010 migration omits `uk_active_relation` unique constraint that `upsert_relation()` relies on | Medium | **Added to migration**: `CREATE UNIQUE INDEX uk_active_relation ON knowledge_relations (project_id, source_entity_id, relation, target_entity_id)` as separate statement in v010. `upsert_relation()` updated to check for invalidated rows and reactivate them (instead of INSERT that would violate the global unique index). |
| 4 | `version` column described as optimistic concurrency but merge UPDATE is unconditional | Medium | **CAS guard added**: `_update_knowledge_with_reembed()` now takes `expected_version` and uses `WHERE id = %s AND version = %s`. Returns `bool` — if CAS fails (concurrent merge incremented version), caller falls through to INSERT. Worst case: a near-duplicate that merges on the next pass. |

### Round 3 (v4)

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | Single `source_knowledge_id` on `knowledge_relations` means last-writer-wins — deleting the last-linked knowledge entry orphans the relation even though other entries still support it | High | **`relation_sources` junction table**: many-to-many between relations and knowledge entries. Each knowledge entry that produces a relation triple is linked via `relation_sources(relation_id, knowledge_id)`. Deleting one knowledge entry removes one source row (CASCADE); the relation stays active as long as any source remains. `gc_unsupported_relations()` invalidates relations with zero remaining sources. |
| 2 | v009 `delete_knowledge_by_source_file()` references `knowledge_relations` table which only exists after v010 — breaks deployment if v009 deployed before v010 | High | ~~v009/v010 independence~~ **Superseded by round 10**: single v008 migration creates all tables before any code references them. No cross-migration ordering issue. |
| 3 | `upsert_relation()` concurrent INSERT unhandled — two writers can race past the SELECT check and both try INSERT, one gets IntegrityError | Medium | **v4: IntegrityError catch with read-after-conflict. v6: superseded by transaction + `SELECT FOR UPDATE`** — all operations in `upsert_relation()` now run within `_pool.transaction()` with a row lock on the source entity, serializing concurrent writers for the same pair. The `uk_relation_tuple` unique index remains as a safety net. |
| OQ | Manual producers (`myswat learn`, `memory add`) don't pass `source_type` to `store_knowledge()` | — | **Phase 1 files-changed updated**: `learn_cmd.py:307,332` and `memory_cmd.py:88` identified as call sites needing `source_type='manual'`. Added to Phase 1 scope. |

### Round 4 (v5)

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | v008-v010 migrations never registered in `schema.py` `MIGRATION_MODULES` — the migration runner only executes modules in that hard-coded list, so none of the new schema changes would ever apply | High | ~~Per-phase schema.py additions~~ **Superseded by round 10**: single `v008_memory_layer` module registered in `MIGRATION_MODULES`. |
| 2 | `upsert_relation()` reactivation path returns early without invalidating conflicting relations — reactivating `(A, uses, B)` can leave `(A, depends_on, B)` active | Medium | **Conflict invalidation moved before existing-tuple check**: the UPDATE that invalidates different-type relations now runs FIRST, before the SELECT for the exact tuple. Reactivation, confidence update, and new INSERT all happen AFTER conflicts are cleared. |
| 3 | `gc_unsupported_relations_if_available()` swallows all exceptions, not just table-not-found — hides real SQL bugs, permissions issues, and transient errors after v010 is deployed | Medium | ~~Narrowed exception~~ **Superseded by round 10**: `gc_unsupported_relations_if_available()` removed entirely. All callers use `gc_unsupported_relations()` directly. |
| OQ | Is `manual` intentionally one source type for both `myswat learn` and ad-hoc `memory add`? Retraction table says `delete_knowledge_by_category("project_ops")` but `memory add` accepts arbitrary categories | — | **Retraction table updated**: `manual` source type covers both producers. `project_ops` category is bulk-retractable via `delete_knowledge_by_category()`; other categories created via `memory add` are individually retractable only (not bulk). |

### Round 5 (v6)

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | Pair-level relation exclusivity not concurrency-safe — two concurrent writers inserting different relation types for the same entity pair can both pass invalidation and both INSERT, leaving two active types. `uk_relation_tuple` only prevents same-tuple duplicates, not cross-type duplicates. | Medium | **Transaction + `SELECT FOR UPDATE`**: `upsert_relation()` now runs entirely within `_pool.transaction()`. Step 0 acquires a row lock on the source entity via `SELECT ... FOR UPDATE`. Concurrent writers for the same entity pair block until the first transaction commits. `TiDBPool.transaction()` and `TiDBTransaction` class added to `db/connection.py`. Phase 3 files-changed updated. Cross-type and same-type concurrent race tests added. |
| OQ | Retraction-unit table says `manual` retracts by `category` but source-type table says only `project_ops` is bulk-retractable | — | **Retraction-unit table updated**: now says `project_ops: bulk by category; other categories: individual only` — consistent with the source-type table. |

### Round 6 (v7)

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | `gc_unsupported_relations()` is racy with concurrent `upsert_relation()` — SELECT of unsupported IDs followed by separate UPDATE allows a concurrent upsert to add a `relation_sources` row between the two steps, causing a now-supported relation to be invalidated | Medium | **Atomic single-statement UPDATE**: replaced two-step SELECT+UPDATE with `UPDATE knowledge_relations SET valid_until = NOW() WHERE ... AND NOT EXISTS (SELECT 1 FROM relation_sources ...)`. The NOT EXISTS subquery is evaluated atomically within the UPDATE, so a relation is only invalidated if it truly has zero sources at execution time. Dry-run mode uses a separate COUNT query. |
| OQ | "Individual delete only" for non-project_ops manual entries has no implementation path — no `delete_knowledge_by_id()` in store.py, no `memory delete` command in CLI | — | **Phase 1 scope updated**: `store.delete_knowledge_by_id()` and `myswat memory delete <id>` command added to Phase 1 files-changed. Both retraction tables now reference the concrete command. |

### Round 7 (v8)

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | Standalone `_add_relation_source()` bypasses the transaction/locking protocol — a caller using it outside `upsert_relation()` could race with `gc_unsupported_relations()`, breaking the "no supported relation gets invalidated" guarantee | Medium | **Removed standalone `_add_relation_source()`**: all `relation_sources` writes are now inlined within `upsert_relation()`'s transaction. No public or internal method exists to add relation sources outside the locking protocol. Design note added: any future batch re-linking must use the same `_pool.transaction()` + entity-row lock pattern. |

### Round 8 (v9)

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | `gc_unsupported_relations()` runs as autocommit outside the entity-row lock protocol — uncommitted `relation_sources` rows from an in-flight `upsert_relation()` transaction are not visible under REPEATABLE READ, so GC can invalidate a relation that a concurrent writer is about to support | Medium | **GC now participates in entity-row locking**: runs within `_pool.transaction()`, acquires `FOR UPDATE` locks on source entity rows of candidate relations (sorted order, no deadlock with single-entity upsert locks), then re-executes the NOT EXISTS UPDATE. After lock acquisition, any concurrent `upsert_relation()` has either committed (rows visible to DML) or is blocked. |
| OQ | `delete_knowledge_by_id()` / `memory delete` doesn't call `gc_unsupported_relations_if_available()` — manual deletion may leave unsupported relations active | — | **`memory delete` command updated**: calls `delete_knowledge_by_id()` then `gc_unsupported_relations()`, same pattern as file retraction in `ingest_file()`. ~~`gc_unsupported_relations_if_available()`~~ removed in round 10. |

### Round 9 (v10)

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | GC step 3 is still a project-wide UPDATE — relations outside the locked candidate set can be invalidated without their entities being locked, so the in-flight visibility problem persists for those unlocked entities | Medium | **Step 3 scoped to candidate IDs**: the UPDATE now uses `WHERE id IN (candidate_ids)` instead of `WHERE project_id = %s`. Only relations whose source entities were locked in step 2 can be invalidated. Relations that became unsupported after step 1 but whose entities were never locked are left for the next GC pass. |

### Round 10 (v11) — No backward compatibility

| # | Change | What was removed |
|---|--------|-----------------|
| 1 | **Single v008 migration** — all schema changes (dedup columns, ingested_documents, entity graph, relation_sources) in one migration version | v009, v010 as separate migrations; "Why three migrations" rationale; deployment-ordering discussion |
| 2 | **Removed `gc_unsupported_relations_if_available()`** — all callers use `gc_unsupported_relations()` directly | Error-1146 try/except wrapper; table-existence checks |
| 3 | **Parser accepts only new format** — `{"knowledge":[], "entities":[]}` required | JSON array fallback (Case 1); backward-compatible dual-format parsing |
| 4 | **Removed `_graph_enabled()` conditional** — graph expansion always runs | Feature-flag/config checks in retriever |
| 5 | **Single `schema.py` entry** — one migration module registered | Per-phase schema.py additions (v008, v009, v010) |
| 6 | **Entity extraction prompt always appended** — no feature flag | Config-gated suffix appending |

### Round 12 (v13) — Historical findings reconciliation

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | "deploys atomically" still present in design body | High | **Already fixed in v12** — GPT reviewed v11, not v12. Grep confirms zero matches for "deploys atomically" in v12+. |
| 2 | Parser call site still shows 2-tuple `items, ok = parse_compaction_output(...)` | Medium | **Already fixed in v12** — call site updated to 3-tuple with entity processing loop. Grep confirms zero matches for old pattern. |
| 3 | "only when entity graph is enabled" contradicts "always appended" | Medium | **Already fixed in v12** — stale gating sentence removed. Grep confirms zero matches. |
| OQ | Historical findings table has stale resolutions (e.g. "Three migrations" in round 1, backward-compatible parser in round 1 OQ, per-phase schema.py in round 4, gc_if_available in round 4/8) | — | **All superseded resolutions struck through** with `~~old text~~` and cross-reference to the round that replaced them (round 10 for all cases). |

### Round 13 (v14) — Entity triple provenance

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | All entity triples linked to `created_ids[-1]` — wrong provenance when compaction produces multiple knowledge items | Medium | **`knowledge_index` field added to entity triple output format**: each triple carries a 0-based index into the `knowledge` array identifying which item it was derived from. Compactor maps `triple["knowledge_index"]` → `created_ids[idx]` for accurate `relation_sources` provenance. Fallback: missing/out-of-range index → `source_knowledge_id=None` (relation created but unprovenanced; `gc_unsupported_relations` will flag it). Prompt suffix updated with the new field spec. |

### Round 14 (v15) — Index-preserving provenance

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | `knowledge_index` lookup broken when items are skipped — `created_ids` is a compressed list (skipped items not represented), so `created_ids[ki]` maps to the wrong ID or goes out of range | Medium | **Replaced `created_ids` list with `kid_by_index` dict**: `kid_by_index[i] = kid` preserves the original array index. Skipped items simply have no entry. Triple lookup uses `kid_by_index.get(ki)` — returns `None` for skipped/missing indices. `created_ids` retained as `list(kid_by_index.values())` for the return value. |

### Round 15 (v16) — Skip unprovenanced triples

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | Triples with missing/invalid `knowledge_index` still create active relations with zero provenance — no GC runs after compaction, so they linger indefinitely | Medium | **Skip instead of create**: provenance check (`kid_by_index.get(ki)`) moved before entity upserts. If `source_kid is None`, the triple is skipped entirely — no entities upserted, no relation created. Unprovenanced relations can never enter the graph from compaction. Comment updated to document the policy. |

---

## How This Closes the Gap with Mem0

| Capability | mem0 | myswat today | myswat after v16 |
|-----------|------|-------------|-----------------|
| Write-time dedup | LLM ADD/UPDATE/DELETE/NOOP | Blind INSERT | Source-scoped `upsert_knowledge()` with vector similarity + LLM merge + embedding recomputation |
| Retraction safety | None (no provenance) | `delete_by_category` only | `source_type` isolation: file/session/manual independently retractable |
| Incremental ingestion | N/A (conversation-only) | Blind re-insert | Content hash tracking, change detection, source-scoped retraction |
| Code awareness | None | 8KB fixed chunks | Rust/Go language-aware chunking at fn/impl boundaries |
| Entity graph | Neo4j/Memgraph | None | `knowledge_entities` + `knowledge_relations` in TiDB with auto-conflict resolution |
| Graph retrieval | Entity-centric + BM25 | Pure vector search | Graph expansion → augmented vector search |
| Deterministic context | None | Tier 0 project ops | Unchanged (better than mem0) |
| Cross-role awareness | None | Tier 1 recent turns | Unchanged (better than mem0) |
| Workflow integration | None | Work items, review cycles | Unchanged (better than mem0) |

---

*Design date: 2026-03-13. v2–v10 revised through rounds 1–9. v11 round 10 (no backward compat). v12 round 11 (atomicity wording, parser call site, stale gating, findings history). v13 round 12 (historical findings reconciliation). v14 round 13 (entity triple provenance via knowledge_index). v15 round 14 (index-preserving kid_by_index dict). v16 round 15 (skip unprovenanced triples).*
